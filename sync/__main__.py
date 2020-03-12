# coding: utf-8


import inspect


def func_signature(func):
    spec = inspect.getargspec(func)
    args = spec.args or []
    defaults = spec.defaults or []
    sig = args[:len(args) - len(defaults)]
    for pair in zip(args[-len(defaults):], defaults):
        sig.append("%s=%s" % pair)
    if spec.varargs:
        sig.append("*" + spec.varargs)
    if spec.keywords:
        sig.append("**" + spec.keywords)
    return sig


def print_usage(funcs):
    print("\n" + " Usage ".center(100, "-") + "\n")

    for func in funcs:
        print("{}({})".format(func.__name__, ", ".join(func_signature(func))))
        if func.__doc__:
            print("    {}".format(func.__doc__.strip()))
        print("")

    print(100 * "-" + "\n")


if __name__ == "__main__":
    import os
    import sys
    import re
    from argparse import ArgumentParser

    # determine directories and files
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sync_dir = os.path.dirname(this_dir)
    default_data_dir = os.path.join(sync_dir, "data")
    default_config = os.path.join(sync_dir, "config", "2018.yml")

    # get arguments
    parser = ArgumentParser(prog="sync", description="synchronization command line tools")
    parser.add_argument("--config", "-c", default=default_config, help="the config file to load, "
        "paths are evaluated relative to the 'config/' directory, default: " + default_config)
    parser.add_argument("--dataset", "-d", help="a dataset to compare, defaults to all")
    parser.add_argument("--data-dir", default=default_data_dir, help="the directory in which "
        "downloaded input files, created tables and plots are stored, default: " + default_data_dir)
    parser.add_argument("--table-format", "-t", default="fancy_grid", help="the tabulate table "
        "format, default: fancy_grid")
    parser.add_argument("--flush", action="store_true", help="flush file cache first")
    parser.add_argument("--ci", action="store_true", help="activate ci mode")
    parser.add_argument("cmd", nargs="*", help="when set, run this command and exit")
    cli = parser.parse_args()

    # import the sync module
    sys.path.insert(0, this_dir)
    import sync

    # sanitize and add some cli args
    cli.config = os.path.join(sync_dir, "config", cli.config)
    cli.cache_dir = os.path.join(cli.data_dir, "input_files")
    cli.table_dir = os.path.join(cli.data_dir, "tables")
    cli.plot_dir = os.path.join(cli.data_dir, "plots")

    # make dirs
    for d in [cli.cache_dir, cli.table_dir, cli.plot_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # create a config object
    print("\nload configuration : {}".format(cli.config))
    config = sync.config.Config.load(cli.config)

    # make certain strings from config global, i.e., accessible without quotes
    config_globals = config.get_globals()
    if config_globals:
        print("global variables   : {}".format(", ".join(config_globals)))
        globals().update({g: g for g in config_globals})

    # create the file cache
    cache = sync.cache.Cache(cli, config)

    # preload all files
    cache.load(flush=cli.flush, ci=cli.ci)

    # set tools globals
    sync.tools.cli = cli
    sync.tools.config = config
    sync.tools.cache = cache

    # import all tools into the local scope
    from sync.tools import *

    # usage helper
    def usage(func=None):
        if func:
            print_usage([func])
        else:
            print_usage([getattr(sync.tools, name) for name in sync.tools.__all__])

    # run the specified sync tool when a command is given, otherwise print usage information
    if cli.cmd:
        # build the command
        if len(cli.cmd) == 1 and re.match(r"^[a-zA-Z0-9_]+\(.*\)$", cli.cmd[0]):
            # trivial case
            cmd = cli.cmd[0]
        else:
            # interpret the first value as a function name and the rest as arguments
            cmd = "{}({})".format(cli.cmd[0], ", ".join(cli.cmd[1:]))

        # run it
        print("running command    : {}\n".format(cmd))
        exec(cmd)
        sys.exit()
    else:
        usage()
