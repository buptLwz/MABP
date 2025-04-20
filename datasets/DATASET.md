## Dataset
We provide the lmdb file of grefcoco that can be downloaded and used directly in <a href="https://pan.baidu.com/s/1eroxdTSnGvihZs293ODEHg?pwd=ysjw" title="model">LMDB</a>

The dataset folder should be like this:

```
datasets
├── lmdb
│   ├── grefcoco
|   |   ├──train.lmdb
|   |   ├──train.lmdb-lock
|   |   ├──val.lmdb
|   |   ├──val.lmdb-lock
|   |   ├──testA.lmdb
|   |   ├──testA.lmdb-lock
|   |   ├──testB.lmdb
|   |   └──testB.lmdb-lock
└── images
    └── train2014
        ├── COCO_train2014_xxxxxxxxxxxx.jpg
        ├── COCO_train2014_xxxxxxxxxxxx.jpg
        └── ...
```

## LMDB

We basically follow the settings of CGFormer (inherited from CRIS) to build the lmdb file, but special modifications are required for grefcoco. Here is the specific process:

1. Download gRefCOCO from [here](https://github.com/henghuiding/gRefCOCO)
2. Ensure that both the annotation and image files are located within the dataset folder
    ```
    datasets
    ├── grefcoco
    │   ├── grefs(unc).json
    │   └── instances.json
    ├── images
        └── train2014
            ├── COCO_train2014_xxxxxxxxxxxx.jpg
            ├── COCO_train2014_xxxxxxxxxxxx.jpg
            └── ...
    └── ...
    ```
3. Generate lmdb files using a workflow similar to CGFormer

    ```# convert
    python ../tools/data_process-gres.py --data_root . --output_dir .   --dataset grefcoco --split unc --generate_mask

    # lmdb
    python ../tools/folder2lmdb.py -j anns/grefcoco/train.json -i    images/train2014/ -m masks/grefcoco -o lmdb/grefcoco
    python ../tools/folder2lmdb.py -j anns/grefcoco/val.json -i  images/train2014/ -m masks/grefcoco -o lmdb/grefcoco
    python ../tools/folder2lmdb.py -j anns/grefcoco/testA.json -i    images/train2014/ -m masks/grefcoco -o lmdb/grefcoco
    python ../tools/folder2lmdb.py -j anns/grefcoco/testB.json -i    images/train2014/ -m masks/grefcoco -o lmdb/grefcoco
    ```

4. [option] If you need to use single-target datasets such as refcoco, you can follow [CRIS](https://github.com/DerrickWang005/CRIS.pytorch/blame/master/tools/prepare_datasets.md) or use ```../tools/data_process.py``` to generate json files in the anns folder, and then use ```folder2lmdb.py``` to create the lmdb files that can be used by MABP.