import argparse
import commands.build

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mycli")
    sub = parser.add_subparsers(dest="command", required=True)
    # Build subcommand
    p_build = sub.add_parser("build", help="build project")
    p_build.add_argument("--in_shape", type=int, default=2)
    p_build.add_argument("--out_shape", type=int, default=1)
    p_build.add_argument("--n_hidden_layers", type=int, default=1)
    p_build.add_argument("--neuron_per_layer", type=int, default=20)
    p_build.add_argument("--actfun", type=str, default="tanh")
    p_build.add_argument("--path", type=str, default="data/")
    # This automatically assigns the function to args.func
    p_build.set_defaults(func=commands.build.cmd_build)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    args.func(args)
