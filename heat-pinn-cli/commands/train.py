import pathlib, joblib, os
import core.train
import pandas as pd
import numpy as np
import tensorflow as tf
import utils.datatools

def cmd_train(args):
    domain = utils.datatools.parquet_to_tensor(args.domain)
    boundary = utils.datatools.parquet_to_tensor(args.boundary)
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
    joblib.dump(trained_model, pathlib.Path("data/models", f"{model.name}-trained.joblib"))
    return 0