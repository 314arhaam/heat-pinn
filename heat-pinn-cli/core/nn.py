import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, List
import os, time, sys, uuid

def dnn_builder(
        in_shape=2, 
        out_shape=1, 
        n_hidden_layers=10,
        neuron_per_layer=20, 
        actfun="tanh"
):
    # input layer
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    # hidden layers
    hidden = [
        tf.keras.layers.Dense(
            neuron_per_layer, 
            activation=actfun
        )(input_layer)
    ]
    for i in range(n_hidden_layers):
        new_layer = tf.keras.layers.Dense(
            neuron_per_layer,
            activation=actfun,
            activity_regularizer=None
        )(hidden[-1])
        hidden.append(new_layer)
    # output layer
    output_layer = tf.keras.layers.Dense(
        out_shape, 
        activation=None
    )(hidden[-1])
    # building the model
    name = f"PINN-DNN-{n_hidden_layers}-{str(uuid.uuid4())}"
    model = tf.keras.Model(input_layer, output_layer, name=name)
    return model