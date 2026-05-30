import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, List
import os, time, sys

@tf.function
def steady_state_2d_heat(u: Callable, x: tf.Tensor, y: tf.Tensor):
    u0 = u(x, y)
    u_x = tf.gradients(u0, x)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    F = u_xx + u_yy
    return tf.reduce_mean(tf.square(F))

@tf.function
def steady_state_3d_heat(u: Callable, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor):
    u0 = u(x, y, z)
    u_x = tf.gradients(u0, x)[0]
    u_y = tf.gradients(u0, y)[0]
    u_z = tf.gradients(u0, z)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    u_zz = tf.gradients(u_z, z)[0]
    F = u_xx + u_yy + u_zz
    return tf.reduce_mean(tf.square(F))

@tf.function
def transient_1d_heat(u: Callable, x: tf.Tensor, t: tf.Tensor):
    u0 = u(x, t)
    u_x = tf.gradients(u0, x)[0]
    u_t = tf.gradients(u0, t)[0]
    u_xx = tf.gradients(u_x, x)[0]
    F = u_xx + u_t
    return tf.reduce_mean(tf.square(F))

@tf.function
def transient_2d_heat(u: Callable, x: tf.Tensor, y: tf.Tensor, t: tf.Tensor):
    u0 = u(x, y, t)
    u_x = tf.gradients(u0, x)[0]
    u_t = tf.gradients(u0, t)[0]
    u_y = tf.gradients(u0, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    F = u_xx + u_yy + u_t
    return tf.reduce_mean(tf.square(F))

@tf.function
def transient_3d_heat(u: Callable, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, t: tf.Tensor):
    u0 = u(x, y, z, t)
    u_x = tf.gradients(u0, x)[0]
    u_t = tf.gradients(u0, t)[0]
    u_y = tf.gradients(u0, y)[0]
    u_z = tf.gradients(u0, z)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    u_zz = tf.gradients(u_z, z)[0]
    F = u_xx + u_yy + u_zz + u_t
    return tf.reduce_mean(tf.square(F))
