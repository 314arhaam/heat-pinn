import argparse

def make_parser() -> argparse.ArgumentParser:
    import commands.build
    import commands.train
    import commands.inference
    parser = argparse.ArgumentParser(prog="mycli")
    sub = parser.add_subparsers(dest="command", required=True)
    # Build subcommand
    p_build = sub.add_parser("build", help="Build PINN")
    p_build.add_argument("--in-shape", type=int, default=2)
    p_build.add_argument("--out-shape", type=int, default=1)
    p_build.add_argument("--n-hidden-layers", type=int, default=1)
    p_build.add_argument("--neuron-per-layer", type=int, default=20)
    p_build.add_argument("--actfun", type=str, default="tanh")
    p_build.add_argument("--name", type=str, default="")
    p_build.add_argument("--path", type=str, default=".")
    p_build.set_defaults(func=commands.build.cmd_build)
    # Train
    p_train = sub.add_parser("train", help="Train PINN")
    p_train.add_argument("--domain", type=str)
    p_train.add_argument("--boundary", type=str)
    p_train.add_argument("--model", type=str)
    # p_train.add_argument("--pde", type=int, default=1)
    # p_train.add_argument("--loss", type=int, default=20)
    p_train.add_argument("--epochs", type=int, default=100)
    # p_train.add_argument("--optimizer", type=str, default="tanh")
    p_train.add_argument("--every", type=int, default=20)
    p_train.set_defaults(func=commands.train.cmd_train)
    # Inference
    p_infer = sub.add_parser("infer", help="PINN Inference")
    p_infer.add_argument("--data", type=str)
    p_infer.add_argument("--model", type=str)
    p_infer.add_argument("--output", type=str, default="inference_result.parquet")
    p_infer.set_defaults(func=commands.inference.cmd_infer)
    return parser

if __name__ == '__main__':
    with open("heat-pinn-cli/banner.txt", "r") as f:
        print(f.read())
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)
