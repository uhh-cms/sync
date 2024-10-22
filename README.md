# Framework synchronization tools

This repository contains tools to synchronize frameworks between groups in terms of event yields, categorization, and object definitions.

## Setup

The setup requires Python ≥3.9 and sets up a virtual environment with all required dependencies.

```shell
source setup.sh
```

Note that a handful of variables that determine installation paths, data paths, etc. can be set a-priori and are not overwritten by the setup script.

## Usage

Start the interactive `sync` tool via:

```shell
sync [arguments]
```

Add `--help` to see all possible arguments.

In general, the tool starts an interactive IPython or Python shell, prints some usage information at startup and waits for input.

The first time you start the tool, it will copy or download the synchronization files of all groups defined by in the configuration file set by `--config FILE` (defaults to [config/2022pre.yml](config/2022pre.yml)).
If you want to reload the files the next time you start the tool, add `--flush`.

As an example, you can show the event yields per category (as defined in the config) and participating group in a certain dataset as demonstrated below.

```shell
> sync

interactive sync tool
(ctrl+d to exit)

....
---------------------------------------------- Usage -----------------------------------------------

usage(func=None) -> None
    Prints usage information of a single *func* when given, or all sync tools otherwise.

print_config() -> None
    Prints a summary of the current configuration.

show_yields(dataset: str | None = None) -> None
    Shows the yields for all groups in a specific *dataset*. When *None*, all datases are
    evaluated sequentially.

----------------------------------------------------------------------------------------------------

In [1]: show_yields()
## Yields for dataset hh

╒════════════════════╤═══════╤═══════╕
│ category / group   │   uhh │   uzh │
╞════════════════════╪═══════╪═══════╡
│ all                │     6 │     6 │
├────────────────────┼───────┼───────┤
│ mutau, OS          │     2 │     2 │
├────────────────────┼───────┼───────┤
│ mutau, SS          │     0 │     0 │
├────────────────────┼───────┼───────┤
│ etau, OS           │     2 │     1 │
├────────────────────┼───────┼───────┤
│ etau, SS           │     0 │     0 │
├────────────────────┼───────┼───────┤
│ tautau, OS         │     2 │     3 │
├────────────────────┼───────┼───────┤
│ tautau, SS         │     0 │     0 │
╘════════════════════╧═══════╧═══════╛

...  # more output
```

## Note on data handling

To keep the repository small, the actual csv input files for comparison should not be committed.
Instead, those files can be located on AFS, Dropbox/CERNBox, etc, which are then copied or downloaded to a local directory the first time the synchronization tool is started.
