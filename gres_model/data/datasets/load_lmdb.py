import logging
import os

"""
This file contains functions to parse RefCOCO-format annotations into dicts in "Detectron2 format".

Replace the load_XX_json here in ReLA with load_lmdb, and shift the task of originally constructing the dict 
to be completed during the construction of lmdb. Thus, a unified load_lmdb can be used to process any dataset. 
"""


logger = logging.getLogger(__name__)

__all__ = ["load_lmdb"]


import lmdb
import pyarrow as pa
def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

def load_lmdb(lmdb_dir):
    env = lmdb.open(lmdb_dir,
                             subdir=os.path.isdir(lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
    with env.begin(write=False) as txn:
        length = loads_pyarrow(txn.get(b'__len__'))
        keys = loads_pyarrow(txn.get(b'__keys__'))

    return keys


