# coding: utf-8

"""
Main entry point of the sync executable.
"""

from __future__ import annotations

import os
import re
import sys
import argparse

import sync
from sync.utils import colored, print_usage


def get_args() -> argparse.Namespace:
    # determine directories and files
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sync_dir = os.path.dirname(this_dir)
    default_config = os.path.join(sync_dir, "config", "2022pre.yml")
    default_output_dir = os.path.expandvars("$SYNC_DATA_DIR/sync_files")

    # get arguments
    parser = argparse.ArgumentParser(
        prog="sync",
        description="synchronization command line tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        default=default_config,
        help="the config file to load, paths are evaluated relative to the 'config/' directory",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="a dataset to compare, defaults to all",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=default_output_dir,
        help="the directory in which to store output files",
    )
    parser.add_argument(
        "--table-format",
        "-t",
        default="fancy_grid",
        help="the tabulate table format",
    )
    parser.add_argument("--flush", action="store_true", help="flush file cache first")
    # parser.add_argument("--ci", action="store_true", help="activate ci mode")
    parser.add_argument("cmd", nargs="*", help="when set, run this command and exit")
    args = parser.parse_args()

    # sanitize and add some args
    args.config = os.path.join(sync_dir, "config", args.config)
    args.cache_dir = os.path.join(args.output_dir, "input_cache")
    args.table_dir = os.path.join(args.output_dir, "tables")
    args.plot_dir = os.path.join(args.output_dir, "plots")

    # make dirs
    for d in [args.cache_dir, args.table_dir, args.plot_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    return args


def main() -> int:
    # get cli arguments
    args = get_args()

    # some header
    print(colored("\ninteractive sync tool", "cyan", style="bright"))
    print("(ctrl+d to exit)\n")

    # create a config object
    print(f"load configuration : {args.config}")
    config = sync.config.Config.from_yaml(args.config)

    # create the data loader
    loader = sync.loader.DataLoader(config, args.cache_dir)

    # make certain strings from config global, i.e., accessible without quotes
    config_globals = config.get_globals()
    if config_globals:
        print(f"global variables   : {','.join(config_globals)}")
        globals().update({g: g for g in config_globals})

    # localize all sync tool methods
    tools = sync.tools.Tools(args, config, loader)
    tool_methods = tools.get_exposed_methods()

    # flush and fetch files
    if args.flush:
        loader.flush()
    loader.fetch()

    # add a usage helper
    def usage(func=None) -> None:
        """
        Prints usage information of a single *func* when given, or all sync tools otherwise.
        """
        if isinstance(func, str):
            func = tool_methods.get(func)
        if func:
            print_usage([func], margin=False)
        else:
            print_usage(list(tool_methods.values()))

    tool_methods = {"usage": usage, **tool_methods}

    # expose all methods
    globals().update(tool_methods)

    # run the specified sync tool when a command is given, otherwise print usage information
    if args.cmd and (args.cmd[0] or len(args.cmd) > 1):
        # build the command
        if len(args.cmd) == 1 and re.match(r"^[a-zA-Z0-9_]+\(.*\)$", args.cmd[0]):
            # trivial case
            cmd = args.cmd[0]
        else:
            # interpret the first value as a function name and the rest as arguments
            cmd = f"{args.cmd[0]}({', '.join(args.cmd[1:])})"

        # run it
        print(f"running command    : {cmd}\n")
        exec(cmd)
        sys.exit()
    else:
        usage()

    return 0


if __name__ == "__main__":
    sys.exit(main())
