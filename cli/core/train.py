import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict
import time
import cli.utils.modeltools

tf.keras.backend.set_floatx("float64")

@tf.function
def f(u: Callable, x: tf.Tensor, y: tf.Tensor):
    u0 = u(x, y)
    u_x = tf.gradients(u0, x)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    F = u_xx + u_yy
    return tf.reduce_mean(tf.square(F))

@tf.function
def mse(y, y_):
    return tf.reduce_mean(tf.square(y-y_))

def train_pinn(
        domain: Dict[str, tf.Tensor],
        boundary: Dict[str, tf.Tensor],
        model: tf.keras.Model,
        loss_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = mse,
        # pde: Callable[[tf.Tensor, tf.Tensor | None, tf.Tensor | None, tf.Tensor | None], tf.Tensor],
        epochs: int = 200, 
        lr: float = 1e-3,
        every: int = 200
) -> Dict[str, Any]:
    optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_values = np.array([])
    u = cli.utils.modeltools.model_wrapper(model)
    #
    start = time.time()
    #
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            T_ = u(boundary['x'], boundary['y'])
            L = f(u, **domain) # pde(**domain)
            l = loss_func(boundary['t'], T_)
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
    return results, model