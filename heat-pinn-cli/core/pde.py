import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, List
import os, time, sys

@tf.function
def f(u: Callable, x: tf.Tensor, y: tf.Tensor):
    u0 = u(x, y)
    u_x = tf.gradients(u0, x)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    F = u_xx + u_yy
    return tf.reduce_mean(tf.square(F))