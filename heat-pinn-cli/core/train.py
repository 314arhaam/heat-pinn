import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, List
import os, time, sys

def _model_wrapper(model):
    @tf.function
    def u(x, y):
        return model(tf.concat([x, y], axis=1))
    return u

def train_pinn(
        domain: Dict[str, tf.Tensor],
        boundary: Dict[str, tf.Tensor],
        model: tf.keras.Model,
        loss_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        pde: Callable[[tf.Tensor, tf.Tensor | None, tf.Tensor | None, tf.Tensor | None], tf.Tensor],
        epochs: int = 200, 
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=5e-4),
        every: int = 200
) -> Dict[str, Any]:
    loss_values = np.array([])
    u = _model_wrapper(model)
    #
    start = time.time()
    #
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            T_ = u(**boundary)
            L = pde(**domain)
            l = loss_func(boundary['value'], T_)
            loss = l+L
        g = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(g, model.trainable_weights))
        loss_values = np.append(loss_values, loss)
        if epoch % every == 0 or epoch == epochs-1:
            print(f"{epoch:5}, {loss.numpy():.3f}")
    #
    end = time.time()
    results = {
        "computation_time": end - start,
        "loss_values": loss_values,
        "model": model,
        "function": u
    }
    return results