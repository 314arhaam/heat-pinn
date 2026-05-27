import tensorflow as tf

tf.keras.backend.set_floatx("float64")

def model_wrapper(model):
    @tf.function
    def u(x, y):
        return model(tf.concat([x, y], axis=1))
    return u