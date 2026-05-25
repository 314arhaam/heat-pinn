import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict

def parquet_to_tensor(path: str):
    data = pd.read_parquet(path)
    tensor_dict = {}
    for col in data.columns:
        tensor_dict[col] = tf.convert_to_tensor(
            np.expand_dims(
                data[col].to_numpy(), 
                axis=1
            ), 
            dtype=tf.float64
        )
    return tensor_dict

def tensor_dict_to_parquet(data: Dict[str, tf.Tensor]):
    data_ = {}
    for key in data.keys():
        data_[key] = data[key].numpy().reshape(-1,)
    return pd.DataFrame(data_)