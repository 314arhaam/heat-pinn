import pathlib, joblib, os
import core.nn

def cmd_build(args):
    in_shape = args.in_shape
    out_shape = args.out_shape
    n_hidden_layers = args.n_hidden_layers
    neuron_per_layer = args.neuron_per_layer
    actfun = args.actfun
    path = args.path
    #
    model = core.nn.dnn_builder(
        in_shape, 
        out_shape, 
        n_hidden_layers,
        neuron_per_layer, 
        actfun
        )
    if args.name == "":
        name = model.name
    else:
        name = args.name
    joblib.dump(model, pathlib.Path(path, f"{name}.joblib"))
    return 0