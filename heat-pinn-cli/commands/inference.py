import joblib
import core.inference
import utils.datatools

def cmd_infer(args):
    data = utils.datatools.parquet_to_tensor(args.data)
    model = joblib.load(args.model)
    output = args.output
    results = core.inference.infer_pinn(
        model=model,
        data=data,
        )
    data["val"] = results
    data = utils.datatools.tensor_dict_to_parquet(data)
    print(data.head())
    data.to_parquet(output)
    return 0