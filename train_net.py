"""
MABP Training Script.

This script is a modified version of the training script in ReLA https://github.com/henghuiding/ReLA.
"""
#python train_net.py --config-file configs/gres-420-1019.yaml  --num-gpus 2 --dist-url auto    MODEL.WEIGHTS /data/lwz/MABP/swin_base_patch4_window12_384_22k.pkl    OUTPUT_DIR try1019
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from functools import reduce
import operator

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    perrankseed_setup,
    launch,
    create_ddp_model
)
from detectron2.evaluation import DatasetEvaluators, verify_results,DataBefore_inference_on_dataset

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from gres_model import (
    ReferEvaluator,
    add_maskformer2_config,
    add_refcoco_config,
    RefCOCOMapperlmdb,
    prepare_data
)
from detectron2.engine.train_loop import AMPTrainer,HookBase
import time
from detectron2.utils.events import EventStorage, get_event_storage
import weakref

from detectron2.evaluation import (
    DatasetEvaluator,
    print_csv_format,
    verify_results,
)
from detectron2.utils.logger import _log_api_usage
torch.set_float32_matmul_precision("high")



class DataBeforeTrainer(AMPTrainer):
    '''We have rewritten the run_step of AMPTrainer here to improve scalability.
    DataBeforeTrainer separate the data preparation (CUDA, normalization, and totensor) in ReLA from model.py
    to provide an interface for future data augmentation and expanding batch sizes during testing'''

    def __init__(self, model, data_loader, optimizer, gather_metric_period=1, 
                 zero_grad_before_forward=False, grad_scaler=None, 
                 precision: torch.dtype = torch.float16, 
                 log_grad_scaler: bool = False, 
                 async_write_metrics=False):
        super().__init__(model, data_loader, optimizer, gather_metric_period, 
                         zero_grad_before_forward, grad_scaler, precision, 
                         log_grad_scaler, async_write_metrics)
        
        if grad_scaler is None:
            from torch.amp import GradScaler
            grad_scaler = GradScaler('cuda')

        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler

    
    def run_step(self):
        """
        Implement the AMP training logic.
        """

        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        
        data = prepare_data(data) # for data preprocessing
        
        data_time = time.perf_counter() - start
        

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        with autocast('cuda',dtype=self.precision):
        
            loss_dict = self.model(data)

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
    
    def reset_data_loader(self, data_loader_builder,cfg):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder(cfg)
        self.data_loader = data_loader
        self._data_loader_iter_obj = None
    

class Trainer(DefaultTrainer):
    '''same as ReLA except self._trainer'''

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)

        self._trainer = DataBeforeTrainer(   # use our modified trainer
            model, data_loader, optimizer
        )
        

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self._trainer.max_iter = self.max_iter

        self.cfg = cfg

        self._period = cfg.TEST.EVAL_PERIOD
        self.register_hooks(self.build_hooks())


    def after_step(self):
    
        for h in self._hooks:
            h.after_step()

        next_iter = self._trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self._trainer.max_iter:
                self._trainer.reset_data_loader(self.build_train_loader,self.cfg)# = self.initfunc(self.cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_list.append(
            ReferEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
                #save_imgs=True     # for visualization, if ture, we provide tools/convert_png.py to transfer the saved .pth to .png
            )
        )
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls,cfg):
        assert cfg.INPUT.DATASET_MAPPER_NAME == "refcoco"
        mapper = RefCOCOMapperlmdb(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        assert cfg.INPUT.DATASET_MAPPER_NAME == "refcoco"
        mapper = RefCOCOMapperlmdb(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, batch_size=1,mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if "text_encoder" in module_name:
                    continue
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0

                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm

                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed


                params.append({"params": [value], **hyperparams})

        hyperparams = copy.copy(defaults)
        params.append({"params": reduce(operator.concat,
                                        [[p for p in model.text_encoder.encoder.layer[i].parameters()
                                          if p.requires_grad] for i in range(10)]), 
                        **hyperparams
                     })

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                #@torch.compile()
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        #optimizer.compile()
        return optimizer
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = DataBefore_inference_on_dataset(model, data_loader, evaluator,logger=logger)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    #default_setup(cfg, args)
    perrankseed_setup(cfg,args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="referring")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
