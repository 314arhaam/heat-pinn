import joblib

def cmd_train(args):
    import cli.core.train
    import cli.utils.datatools
    domain = cli.utils.datatools.parquet_to_tensor(args.domain)
    boundary = cli.utils.datatools.parquet_to_tensor(args.boundary)
    model = joblib.load(args.model)
    # loss
    epochs = args.epochs
    every = args.every
    #
    results, trained_model = cli.core.train.train_pinn(
        domain=domain,
        boundary=boundary,
        model=model,
        epochs=epochs,
        every=every
        )
    print(results)
    joblib.dump(trained_model, args.model)
    return 0