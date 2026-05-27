import tensorflow as tf
from typing import Dict
import cli.utils.modeltools

tf.keras.backend.set_floatx("float64")

def infer_pinn(model: tf.keras.Model, data: Dict[str, tf.Tensor]):
    u = cli.utils.modeltools.model_wrapper(model)
    return u(**data)