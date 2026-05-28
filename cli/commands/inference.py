import joblib

def cmd_infer(args):
    import cli.core.inference
    import cli.utils.datatools
    data = cli.utils.datatools.parquet_to_tensor(args.data)
    model = joblib.load(args.model)
    output = args.output
    results = cli.core.inference.infer_pinn(
        model=model,
        data=data,
        )
    data["val"] = results
    data = cli.utils.datatools.tensor_dict_to_parquet(data)
    print(data.head())
    data.to_parquet(output)
    return 0