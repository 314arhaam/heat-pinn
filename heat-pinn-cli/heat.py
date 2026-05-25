import argparse

def make_parser() -> argparse.ArgumentParser:
    import commands.build
    import commands.train
    parser = argparse.ArgumentParser(prog="mycli")
    sub = parser.add_subparsers(dest="command", required=True)
    # Build subcommand
    p_build = sub.add_parser("build", help="build project")
    p_build.add_argument("--in-shape", type=int, default=2)
    p_build.add_argument("--out-shape", type=int, default=1)
    p_build.add_argument("--n-hidden-layers", type=int, default=1)
    p_build.add_argument("--neuron-per-layer", type=int, default=20)
    p_build.add_argument("--actfun", type=str, default="tanh")
    p_build.add_argument("--path", type=str, default="data/")
    p_build.set_defaults(func=commands.build.cmd_build)
    # Train
    p_build = sub.add_parser("train", help="train PINN")
    p_build.add_argument("--domain", type=str)
    p_build.add_argument("--boundary", type=str)
    p_build.add_argument("--model", type=str)
    # p_build.add_argument("--pde", type=int, default=1)
    # p_build.add_argument("--loss", type=int, default=20)
    p_build.add_argument("--epochs", type=int, default=100)
    # p_build.add_argument("--optimizer", type=str, default="tanh")
    p_build.add_argument("--every", type=int, default=20)
    p_build.set_defaults(func=commands.train.cmd_train)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    args.func(args)
