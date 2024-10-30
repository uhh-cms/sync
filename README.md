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

---------------------------------------------- Usage -----------------------------------------------

usage(func=None) -> None
    Prints usage information of a single *func* when given, or all sync tools otherwise.

check_common_events(dataset: str, groups: str | None = None, variables: str | None = None, interactive: bool = True) -> None
    Traverses events in a *dataset* that are common to all *groups* and prints a table with
    specific *variables* per event. When *groups* is *None*, all participating groups are
    selected. When *variables* is *None*, the variables defined in the configuration are used.
    In case of multiple events, a prompt allows to either stop or continue the comparison when
    *interactive* is *True*.

check_missing_events(dataset: str, group1: str, group2: str, variables: str | None = None, interactive: bool = True) -> None
    Traverses missing events between *group1* and *group2* in a specific *dataset* and prints a
    table with specific *variables* per event. When *variables* is *None*, all variables defined
    in the configuration are used. In case of multiple events, a prompt allows to either stop or
    continue the comparison when *interactive* is *True*.

compare_event(dataset: str, event: int | None = None, variables: str | None = None, interactive: bool = True) -> None
    Compares *variables* of an *event* given by its id in a specific *dataset*. When *variables*
    is None*, the variables defined in the configuration are used. When *event* is *None*, all
    events in that dataset compared. In case of multiple events, a prompt allows to either stop
    or continue the comparison when *interactive* is *True*.

compare_variable(dataset: str, variable: str, group1: str, group2: str, epsilon: float = 1e-05) -> None
    Compares a *variable* in a specific *dataset* between *group1* and *group2* and prints a
    table showing variable values in differing events, i.e., in events where the relative
    difference exceeds *epsilon*.

compare_yields(dataset: str, group1: str, group2: str) -> None
    Compares the yields in a specific *dataset* between *group1* and *group2*, subdivided into
    all known categories.

draw_variable(dataset: str | None = None, variable: str | None = None, group: str = None, bins: int = 20) -> None
    Creates a histogram with number of *bins* also including a ratio relative to *group*.
    The plot is created specific for a *dataset* and *variable* and saves it in the plot
    directory. When *dataset* is *None*, all available datasets are used. When *variables* is
    *None*, the variables defined in the configuration are used.

print_config() -> None
    Prints a summary of the current configuration.

show_yields(dataset: str | None = None) -> None
    Shows the yields for all groups in a specific *dataset*. When *None*, all datases are
    evaluated sequentially.

visualize_variable(dataset: str | None = None, variables: str | None = None, epsilon: float = 1e-05) -> None
    Creates a visualization for a specific *dataset* and *variables* and saves it in the plot
    directory. When *dataset* is *None*, all available datasets are used. When *variables* is
    *None*, the variables defined in the configuration are used.

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
