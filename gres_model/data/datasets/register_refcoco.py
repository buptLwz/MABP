import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .load_lmdb import load_lmdb

'''We provide LMDB files for refcoco/+/g, and you can follow the example of grefcoco to register them accordingly.'''

def register_grefcoco_lmdb(root):
    lmdb_root = os.path.join(root, "lmdb", "grefcoco")
    dataset_info = [
        (('grefcoco'), 'unc', ['train.lmdb','val.lmdb','testA.lmdb','testB.lmdb']),
    ]
    for name_list, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name_list, splitby, split])
            DatasetCatalog.register(
                dataset_id,
                lambda root=os.path.join(lmdb_root,split): 
                    load_lmdb(root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name_list,
                splitby=splitby,
                split=split,
                root=root,
                image_root=root,
            )

def register_mevis_lmdb(root):
    lmdb_root = os.path.join(root, "lmdb", "mevis")
    dataset_info = [
        (('mevis'), 'unc', ['train.lmdb','valid-u.lmdb']),
    ]
    for name_list, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name_list, splitby, split])
            DatasetCatalog.register(
                dataset_id,
                lambda root=os.path.join(lmdb_root,split): 
                    load_lmdb(root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name_list,
                splitby=splitby,
                split=split,
                root=root,
                image_root=root,
            )


def register_refzom_lmdb(root):
    lmdb_root = os.path.join(root, "lmdb", "refzom")
    dataset_info = [
        (('refzom'), 'final', ['train.lmdb','test.lmdb']),
    ]
    for name_list, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name_list, splitby, split])
            DatasetCatalog.register(
                dataset_id,
                lambda root=os.path.join(lmdb_root,split): 
                    load_lmdb(root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name_list,
                splitby=splitby,
                split=split,
                root=root,
                image_root=root,
            )

def register_rrsisd_lmdb(root):
    lmdb_root = os.path.join(root, "lmdb", "rrsisd")
    dataset_info = [
        (('rrsisd'), 'unc', ['train.lmdb','val.lmdb']),
    ]
    for name_list, splitby, splits in dataset_info:
        for split in splits:
            dataset_id = '_'.join([name_list, splitby, split])
            DatasetCatalog.register(
                dataset_id,
                lambda root=os.path.join(lmdb_root,split): 
                    load_lmdb(root)
            )
            MetadataCatalog.get(dataset_id).set(
                evaluator_type="refer",
                dataset_name=name_list,
                splitby=splitby,
                split=split,
                root=root,
                image_root=root,
            )

# regisiter dataset 
_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_grefcoco_lmdb(_root)
register_refzom_lmdb(_root)
register_rrsisd_lmdb(_root)
register_mevis_lmdb(_root)

