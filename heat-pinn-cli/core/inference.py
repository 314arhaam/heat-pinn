import tensorflow as tf
from typing import Dict

tf.keras.backend.set_floatx("float64")

def _model_wrapper(model):
    @tf.function
    def u(x, y):
        return model(tf.concat([x, y], axis=1))
    return u

def infer_pinn(model: tf.keras.Model, data: Dict[str, tf.Tensor]):
    u = _model_wrapper(model)
    return u(**data)