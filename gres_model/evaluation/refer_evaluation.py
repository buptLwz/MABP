import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import torch

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

class ReferEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        save_imgs=False,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._save_imgs = save_imgs

        self._cpu_device = torch.device("cpu")
        self._available_sources = ["refcoco", "grefcoco"]

        self._num_classes = 2

    def reset(self):
        self._predictions = []
        
    def process(self, inputs, outputs):
        assert len(inputs)==1
        gt_mask = inputs[0]['mask'].to(self._cpu_device)
        for output in outputs:

            # output mask
            output_mask = (output["pred_mask"]>0.5).to(self._cpu_device)
            pred_mask = np.array(output_mask, dtype=np.int8)

            gt = np.array(gt_mask, dtype=np.int8)
            
            output_nt = [(i.argmax(dim=0)).bool().to(self._cpu_device) for i in output["pred_nt"]]

            '''Trick: perform a union operation on all intermediate results to enhance the recall for non-target samples.
            This achieves better results compared to those in the paper.
            '''
            output_nt = any(output_nt) 
            

            pred_nt = bool(output_nt)

            self._predictions.append({
                'img_name': inputs[0]['img_name'],
                'segment_id':inputs[0]['segment_id'],
                'sent': output['sent'],
                'pred_nt': pred_nt,
                'empty': bool(inputs[0].get('empty', False)[0]),
                'pred_mask': pred_mask, 
                'gt_mask': gt,
                })

    def evaluate(self):
        '''same as ReLA'''
        if self._distributed:
            synchronize()
            predictions = all_gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
            if not is_main_process():
                return
        else:
            predictions = self._predictions
        
        if self._output_dir and self._save_imgs:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "ref_seg_predictions.pth")
            self._logger.info(f'Saving output images to {file_path} ...')
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
        
        pr_thres = [.7, .8, .9]

        accum_I = {}
        accum_U = {}
        accum_IoU = {}
        pr_count = {}
        total_count = {}
        not_empty_count = {}
        empty_count = {}
        nt = {}

        for src in self._available_sources:
            accum_I[src] = 0
            accum_U[src] = 0
            accum_IoU[src] = 0

            pr_count[src] = {}
            for thres in pr_thres:
                pr_count[src][thres] = 0

            total_count[src] = 0
            not_empty_count[src] = 0
            empty_count[src] = 0
            nt[src] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        results_dict = []

        for eval_sample in predictions:

            ref_result = {}
            ref_result['source'] = src
            ref_result['img_name'] = eval_sample['img_name']
            ref_result['empty'] = eval_sample['empty']
            ref_result['pred_nt'] = eval_sample['pred_nt']
            ref_result['sent'] = eval_sample['sent']

            I, U = computeIoU(eval_sample['pred_mask'], eval_sample['gt_mask'])

            # No-target Samples
            if eval_sample['empty']:
                empty_count[src] += 1
                ref_result['I'] = int(0)

                # True Positive
                if eval_sample['pred_nt']:
                    nt[src]["TP"] += 1
                    accum_IoU[src] += 1
                    accum_I[src] += 0
                    accum_U[src] += 0

                    ref_result['U'] = int(0)
                    ref_result['cIoU'] = float(1)

                # False Negative
                else:
                    nt[src]["FN"] += 1   #FP
                    accum_IoU[src] += 0
                    accum_I[src] += 0
                    accum_U[src] += int(U)

                    ref_result['U'] = int(U)
                    ref_result['cIoU'] = float(0)

            # Targeted Samples
            else:
                # False Positive
                if eval_sample['pred_nt']:  #FN
                    nt[src]["FP"] += 1
                    I = 0

                # True Negative
                else:                   #TN
                    nt[src]["TN"] += 1

                this_iou = float(0) if U == 0 else float(I) / float(U)

                accum_IoU[src] += this_iou
                accum_I[src] += I
                accum_U[src] += U

                not_empty_count[src] += 1

                for thres in pr_thres:
                    if this_iou >= thres:
                        pr_count[src][thres] += 1

                ref_result['I'] = int(I)
                ref_result['U'] = int(U)
                ref_result['cIoU'] = float(this_iou)

            total_count[src] += 1
            results_dict.append(ref_result)

        detected_srcs = [src for src in self._available_sources if total_count[src] > 0]

        final_results_list = []
        
        # results for each source
        for src in detected_srcs:
            res = {}
            res['gIoU'] = 100. * (accum_IoU[src] / total_count[src])
            res['cIoU'] = accum_I[src] * 100. / accum_U[src]

            self._logger.info(str(nt[src]))
            if empty_count[src] > 0:
                res['T_acc'] = nt[src]['TP'] / (nt[src]['TP'] + nt[src]['FP'] + 1)
                res['N_acc'] = nt[src]['TP'] / (nt[src]['TP'] + nt[src]['FN'] + 1)
            else:
                res['T_acc'] = res['N_acc'] = 0
            if not_empty_count[src]==0:
                not_empty_count[src]=1
            for thres in pr_thres:
                pr_name = 'Pr@{0:1.1f}'.format(thres)
                res[pr_name] = pr_count[src][thres] * 100. / not_empty_count[src]
            
            final_results_list.append((src, res))
        
        def _sum_values(x):
            return sum(x.values())
        
        # global results
        if len(detected_srcs) > 1:
            res_full = {}
            res_full['gIoU'] = 100. * _sum_values(accum_IoU) / _sum_values(total_count)
            res_full['cIoU'] =  100. * _sum_values(accum_I) / _sum_values(accum_U)

            for thres in pr_thres:
                pr_name = 'Pr@{0:1.1f}'.format(thres)
                res_full[pr_name] = sum([pr_count[src][thres] for src in detected_srcs]) * 100. / _sum_values(not_empty_count)

            final_results_list.append(('full', res_full))
        
        if self._output_dir:
            file_path = os.path.join(self._output_dir, f"{self._dataset_name}_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(final_results_list, indent=4))   

            file_path = os.path.join(self._output_dir, f"{self._dataset_name}_detailed_results.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(results_dict, indent=4))             

        results = OrderedDict(final_results_list)
        self._logger.info(results)
        return results
