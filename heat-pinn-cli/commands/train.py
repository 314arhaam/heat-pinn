import pathlib, joblib, os
import core.train
import pandas as pd
import numpy as np
import tensorflow as tf

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

def cmd_train(args):
    domain = parquet_to_tensor(args.domain)
    boundary = parquet_to_tensor(args.boundary)
    model = joblib.load(args.model)
    # loss
    epochs = args.epochs
    every = args.every
    #
    results, trained_model = core.train.train_pinn(
        domain=domain,
        boundary=boundary,
        model=model,
        epochs=epochs,
        every=every
        )
    print(results)
    joblib.dump(trained_model, pathlib.Path("data", f"{model.name}-trained.joblib"))
    return 0