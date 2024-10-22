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

As an example, you show the event yields per category (also defined in the config) and participating group in a certain dataset.
The full output will look like the following.

```shell
> sync

interactive sync tool
(ctrl+d to exit)

tba
```

<!--
---------------------------------------------- Usage -----------------------------------------------

print_config()
    Prints a summary of the current configuration.

show_yields(dataset=None)
    Shows the yields for all groups in a specific *dataset*. When *None*, all datases are evaluated
    sequentially.

write_yields(dataset=None)
    Writes the yield tables obtained by :py:func:`show_yields` into a file per dataset in the
    table directory. When *dataset* is *None*, all datasets are evaluated sequentially.

...  # more tools

----------------------------------------------------------------------------------------------------

In [1]: show_yields()
## Yields for dataset data_mu

╒════════════════════╤═════════╤═════════╤═════════╕
│ category / group   │ Group X │ Group Y │ ...     │
╞════════════════════╪═════════╪═════════╪═════════╡
│ ≥2j ≥1b            │     180 │     180 │     ... │
├────────────────────┼─────────┼─────────┤─────────┤
│ ≥1j ≥1b            │     225 │     225 │     ... │
├────────────────────┼─────────┼─────────┤─────────┤
│ ...                │     ... │     ... │     ... │
╘════════════════════╧═════════╧═════════╛═════════╛

...  # more output
```
-->

## Note on data handling

To keep the repository small, the actual csv input files for comparison should not be committed.
Instead, those files can be located on AFS, Dropbox/CERNBox, etc, which are then copied or downloaded to a local directory the first time the synchronization tool is started.
