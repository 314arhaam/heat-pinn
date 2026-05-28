import argparse

def make_parser() -> argparse.ArgumentParser:
    import cli.commands.build
    import cli.commands.train
    import cli.commands.inference
    parser = argparse.ArgumentParser(prog="heat")
    sub = parser.add_subparsers(dest="command", required=True)
    # Build subcommand
    p_build = sub.add_parser("build", help="Build PINN")
    p_build.add_argument(
        "--in-shape", 
        type=int, 
        default=2, 
        help="Shape of the input tensor to feed into NN. Equal to the number of independent variables of PDE."
    )
    p_build.add_argument(
        "--out-shape", 
        type=int, 
        default=1,
        help="Shape of the output tensor of NN. For heat transfer it's T (equal to 1)"
    )
    p_build.add_argument(
        "--n-hidden-layers", 
        type=int, 
        default=1,
        help="Number of hidden layers in the NN"
    )
    p_build.add_argument(
        "--neuron-per-layer", 
        type=int, 
        default=20,
        help="Number of neurons in each hidden layer"
    )
    p_build.add_argument(
        "--actfun", 
        type=str, 
        default="tanh",
        help="Activation function"
    )
    p_build.add_argument(
        "--name", 
        type=str, 
        default="",
        help="Name of the model."
    )
    p_build.set_defaults(func=cli.commands.build.cmd_build)
    # Train
    p_train = sub.add_parser("train", help="Train PINN")
    p_train.add_argument(
        "--domain", 
        type=str, 
        help="Path of domain data file"
    )
    p_train.add_argument(
        "--boundary", 
        type=str,
        help="Path of boundary data file"
    )
    p_train.add_argument(
        "--model", 
        type=str,
        help="Path of model file"
    )
    # p_train.add_argument("--pde", type=int, default=1)
    # p_train.add_argument("--loss", type=int, default=20)
    p_train.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    # p_train.add_argument("--optimizer", type=str, default="tanh")
    p_train.add_argument(
        "--every", 
        type=int, 
        default=20,
        help="Print result for every n epochs"
    )
    p_train.set_defaults(func=cli.commands.train.cmd_train)
    # Inference
    p_infer = sub.add_parser("infer", help="PINN Inference")
    p_infer.add_argument(
        "--data", 
        type=str,
        help="Path of data file to perform inference"
    )
    p_infer.add_argument(
        "--model", 
        type=str,
        help="Path of model file"
    )
    p_infer.add_argument(
        "--output", 
        type=str, 
        default="inference_result.parquet",
        help="Path of output data file"
    )
    p_infer.set_defaults(func=cli.commands.inference.cmd_infer)
    return parser

def main():
    with open("cli/banner.txt", "r") as f:
        print(f.read())
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
