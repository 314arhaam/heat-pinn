import pathlib, joblib, os
import core.nn

def cmd_build(args):
    in_shape = args.in_shape
    out_shape = args.out_shape
    n_hidden_layers = args.n_hidden_layers
    neuron_per_layer = args.neuron_per_layer
    actfun = args.actfun
    #
    model = core.nn.dnn_builder(
        in_shape, 
        out_shape, 
        n_hidden_layers,
        neuron_per_layer, 
        actfun
        )
    if args.name == "":
        name = f"{model.name}.joblib"
    else:
        name = f"{args.name}"
    joblib.dump(model, f"{name}")
    return 0