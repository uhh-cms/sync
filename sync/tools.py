# coding: utf-8

"""
Synchronization tools.
"""


__all__ = [
    "print_config", "show_yields", "write_yields", "compare_yields", "compare_event", "write_event",
    "check_missing_events", "check_common_events", "compare_variable", "visualize_variable",
    "write_all",
]


import os
import sys
import math
import contextlib
import collections
import itertools


# global variables, to be set externally
cli = None
config = None
cache = None

MISSING = -999.


#
# helpers
#

@contextlib.contextmanager
def change_stdout(f):
    orig_stdout = sys.stdout
    sys.stdout = f
    try:
        yield
    finally:
        sys.stdout = orig_stdout


def print_table(*args, **kwargs):
    import six
    import tabulate

    kwargs.setdefault("tablefmt", cli.table_format)
    table = tabulate.tabulate(*args, **kwargs)

    print(table.encode("utf-8") if six.PY2 else table)


def get_datasets(dataset=None):
    if dataset is None:
        return list(config.get_datasets())
    elif isinstance(dataset, list):
        return list(dataset)
    else:
        return [dataset]


def get_variables(variable=None):
    if not variable:
        return list(config.get_variables())
    elif isinstance(variable, list):
        return list(variable)
    else:
        return [variable]


def draw_variable_comparison(dataset, variable, diffs, path, epsilon=1e-5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    # plot title
    title = "Dataset '{}', Variable '{}'".format(dataset, variable)

    # diffs is a dict with keys being 2-tuples of groups,
    # extract sorted groups
    groups = sorted(list(set(sum((list(tpl) for tpl in diffs.keys()), []))))

    # create a networkx graph
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    graph = nx.Graph()

    # helper to compute the rotation for group i
    def i2phi(i):
        # add 90 deg offset to start at the top
        return i * 2 * math.pi / len(groups) + math.pi / 2.

    # helper to translate from polar to cartesian coordinates
    def p2c(r, phi, y_offset=0.):
        return (r * math.cos(phi), r * math.sin(phi) + y_offset)

    # prepare graph attributes
    node_pos = {}
    label_pos = {}
    labels = {}
    for i, group in enumerate(groups):
        # y offset to align the shape vertically
        y_offset = -0.5 * (1 + math.sin(i2phi(len(groups) // 2)))
        # calculate positions
        phi = i2phi(i)
        node_pos[i] = p2c(0.8, phi, y_offset)
        label_pos[i] = p2c(0.95, phi, y_offset)
        labels[i] = group
        graph.add_node(i, pos=node_pos[i])

    # draw nodes and labels
    nx.draw(graph, node_pos, node_color="blue")
    nx.draw_networkx_labels(graph, label_pos, labels, font_size=14)
    nx.draw_networkx_labels(graph, {0: p2c(1.1, math.pi / 2.)}, {0: title}, font_size=18)

    # add edges, based on differences
    for (group1, group2), _diffs in diffs.items():
        # compile the differences into a single value
        # method 1: take the absolute variance of non-zero values
        idxs = _diffs != 0
        diff = abs(_diffs[_diffs != 0].var()) if idxs.any() else 0.
        # method 2: take the std of non-zero values
        # diff = _diffs[_diffs != 0].std() if idxs.any() else 0.

        # define a line weight between 1 and 10 by using an activation-like approach
        # see https://www.wolframalpha.com/input/?i=plot+10+*+tanh%28x+*+2%29+for+x%3D0+to+1.
        weight = 2 if diff <= epsilon else max(1, 10 * math.tanh(diff * 2))

        # define a line color
        color = "green" if diff <= epsilon else "red"

        nx.draw_networkx_edges(graph, node_pos, [(groups.index(group1), groups.index(group2))],
            width=weight, edge_color=color)

    # save it
    print("save comparison plot of variable {} in dataset {} at {}".format(variable, dataset, path))
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.close()


#
# actual sync tools
#

def print_config():
    """
    Prints a summary of the current configuration.
    """
    # datasets and groups
    print("datasets and participating groups:")
    for dataset in config.get_datasets():
        print("    dataset: {}".format(dataset))
        for group in config.get_groups(dataset):
            print("        group: {}".format(group))

    # variables
    print("\nvariables:")
    for name in config.get_variables():
        print("    - {}".format(name))

    # categories
    print("\ncategories:")
    for name, expr in config.get_categories().items():
        print("    - {}: {}".format(name, expr))


def show_yields(dataset=None):
    """
    Shows the yields for all groups in a specific *dataset*. When *None*, all datases are evaluated
    sequentially.
    """
    # default datasets
    datasets = get_datasets(dataset)

    def show(dataset):
        print("## Yields for dataset {}\n".format(dataset))

        groups = config.get_groups(dataset)
        headers = ["category / group"] + groups
        table = []

        for cat, cat_expr in config.get_categories().items():
            line = [cat]
            for group in groups:
                df = cache.get(dataset, group)
                try:
                    line.append(sum(df.eval(cat_expr)))
                except ImportError:
                    raise
                except Exception as e:
                    e.message = "evaluation failed for group {}, dataset {}, category {}\n{}".format(
                        group, dataset, cat, e.message)
                    raise

            table.append(line)

        print_table(table, headers=headers)

    for dataset in datasets:
        show(dataset)
        print("")


def write_yields(dataset=None):
    """
    Writes the yield tables obtained by :py:func:`show_yields` into a file per dataset in the
    table directory. When *dataset* is *None*, all datasets are evaluated sequentially.
    """
    datasets = get_datasets(dataset)

    for dataset in datasets:
        path = os.path.join(cli.table_dir, "yields__{}.md".format(dataset))
        if os.path.exists(path):
            os.remove(path)

        print("write yields for dataset {} to {}".format(dataset, path))

        with open(path, "w") as f:
            with change_stdout(f):
                show_yields(dataset)


def compare_yields(dataset, group1, group2):
    """
    Compares the yields in a specific *dataset* between *group1* and *group2*.
    """
    print("## Yield comparison for dataset {} between {} and {}\n".format(dataset, group1, group2))

    headers = ["category / type", group1, group2, "common", "{} - {}".format(group1, group2),
        "{} - {}".format(group2, group1)]
    table = []

    df1 = cache.get(dataset, group1)
    df2 = cache.get(dataset, group2)
    for cat, cat_expr in config.get_categories().items():
        _df1 = df1[df1.eval(cat_expr)]
        _df2 = df2[df2.eval(cat_expr)]

        # get sets of the event id for simple comparison
        s1 = set(_df1["event"].values)
        s2 = set(_df2["event"].values)

        table.append((cat, len(s1), len(s2), len(s1 & s2), len(s1 - s2), len(s2 - s1)))

    print_table(table, headers=headers)


def compare_event(dataset, event=None, variables=None, interactive=True):
    """
    Compares *variables* of an *event* given by its id in a specific *dataset*. When *variables* is
    *None*, the variables defined in the configuration are used. When *event* is *None*, all events
    in that dataset compared. In case of multiple events, a prompt allows to either stop or continue
    the comparison when *interactive* is *True*.
    """
    # get participating groups
    groups = config.get_groups(dataset)

    # default variables
    variables = get_variables(variables)

    # default events
    if not event:
        events = list(set.union(*(set(cache.get(dataset, g)["event"].values) for g in groups)))
    elif isinstance(event, list):
        events = list(event)
    else:
        events = [event]

    def compare(event):
        print("\n## Comparison of event {} in dataset {}\n".format(event, dataset))

        selection = "(event == {})".format(event)
        table = [[group] for group in groups]

        for group, row in zip(groups, table):
            df = cache.get(dataset, group)
            idxs = df.eval(selection)
            n = sum(idxs)
            if n == 0:
                # fill some missing value
                row.extend(MISSING for _ in variables)
            else:
                if n != 1:
                    raise Exception("event {} contained {} times in dataset {} of group {}".format(
                        event, n, dataset, group))
                for v in variables:
                    val = df[idxs][v].values[0]
                    if isinstance(val, int) or val == -1:
                        val = str(int(val))  # numpy-safe conversion
                    row.append(val)

        print_table(table, headers=["group"] + list(variables), floatfmt=".4f")

    # loop over events
    for i, event in enumerate(events):
        compare(event)
        if i < len(events) - 1:
            print("")
            if interactive:
                next_event = events[i + 1]
                inp = raw_input("press enter to continue with next event ({}) or any other key to "
                    "stop: ".format(next_event)).strip()
                if inp:
                    break


def write_event(dataset, event=None, variables=None):
    """
    Writes the event comparison tables obtained by :py:func:`compare_event` into a file for a
    specific *dataset* and *variables* for a selected *event*. When *None*, the *test_events* list
    in the configuration entry for that dataset is used. *variables* is forwarded to
    :py:func:`compare_event`.
    """
    # default events
    if not event:
        events = config["datasets"][dataset]["test_events"]

    path = os.path.join(cli.table_dir, "events__{}.md".format(dataset))
    if os.path.exists(path):
        os.remove(path)

    print("write event comparison for dataset {} to {}".format(dataset, path))

    with open(path, "w") as f:
        with change_stdout(f):
            compare_event(dataset, events, variables=variables, interactive=False)


def check_missing_events(dataset, group1, group2, variables=None, interactive=True):
    """
    Traverses missing events between *group1* and *group2* in a specific *dataset* and prints a
    table with specific *variables* per event. When *variables* is *None*, the variables defined in
    the configuration are used.
    """
    print("## Missing events for dataset {} between {} and {}\n".format(dataset, group1, group2))

    # default variables
    variables = get_variables(variables)

    # get data frames
    df1 = cache.get(dataset, group1)
    df2 = cache.get(dataset, group2)

    # create event set differences
    s1 = set(df1["event"].values)
    s2 = set(df2["event"].values)
    diff12 = s1 - s2
    diff21 = s2 - s1

    print("### Stats and differences\n")
    print("{}: {} events".format(group1, len(s1)))
    print("{}: {} events".format(group2, len(s2)))
    print("{} - {}: {} events".format(group1, group2, len(diff12)))
    print("{} - {}: {} events".format(group2, group1, len(diff21)))
    print("{} | {}: {} events".format(group1, group2, len(s1 | s2)))
    print("{} & {}: {} events".format(group1, group2, len(s1 & s2)))

    def traverse_diff(group1, group2, df, diff, can_reverse=False):
        print("\n### Traversing {} - {}\n".format(group1, group2))

        print("missing: {}\n".format(",".join(str(e) for e in diff)))

        for event in diff:
            headers = ["# {}".format(event)] + list(variables)
            row = [group1]

            idxs = df.eval("(event == {})".format(event))
            if idxs.sum() != 1:
                raise Exception("event {} contained {} times in dataset {} of group {}".format(
                    event, len(idxs), dataset, group1))
            row.extend(df[idxs][v].values[0] for v in variables)

            print_table([row], headers=headers, floatfmt=".4f")

            if interactive:
                print("")
                if can_reverse:
                    inp = raw_input("press enter to continue, 'r' to reverse groups, or any other "
                        "key to stop: ").strip()
                    if inp.lower() == "r":
                        return True
                    elif inp:
                        return False
                else:
                    inp = raw_input("press enter to continue, or any other key to stop: ").strip()
                    if inp:
                        return False
            print("")
        return True

    # traverse diffs, based on which group misses events
    if len(diff12) and len(diff21):
        # both groups miss some events that the other group has
        # so allow for switching what set to traverse
        if traverse_diff(group1, group2, df1, diff12, can_reverse=True):
            print("")
            traverse_diff(group2, group1, df2, diff21)
    elif len(diff12):
        traverse_diff(group1, group2, df1, diff12)
    elif len(diff21):
        traverse_diff(group2, group1, df2, diff21)


def check_common_events(dataset, groups=None, variables=None, interactive=True):
    """
    Traverses events in a *dataset* that are common to all *groups* and prints a table with specific
    *variables* per event. When *groups* is *None*, all participating groups are selected. When
    *variables* is *None*, the variables defined in the configuration are used. In case of multiple
    events, a prompt allows to either stop or continue the comparison when *interactive* is *True*.
    """
    # default groups
    if not groups:
        groups = config.get_groups(dataset)

    # default variables
    variables = get_variables(variables)

    print("## Common events for dataset {} between {}\n".format(dataset, ", ".join(groups)))

    # get common events
    common = set()
    for group in groups:
        df = cache.get(dataset, group)
        s = set(df["event"].values)
        common = s if not common else set.intersection(common, s)

    # traverse common events
    for i, event in enumerate(common):
        headers = ["# {}".format(event)] + list(variables)
        table = []

        selection = "(event == {})".format(event)
        for group in groups:
            df = cache.get(dataset, group)
            idxs = df.eval(selection)
            if idxs.sum() != 1:
                raise Exception("event {} contained {} times in dataset {} of group {}".format(
                    event, len(idxs), dataset, group))
            row = [group] + [df[idxs][v].values[0] for v in variables]
            table.append(row)

        print_table(table, headers=headers, floatfmt=".4f")

        if i < len(common) - 1:
            print("")
            if interactive:
                inp = raw_input("press enter to continue, or any other key to stop: ").strip()
                if inp:
                    break
                print("")


def compare_variable(dataset, variable, group1, group2, epsilon=1e-5):
    """
    Compares a *variable* in a specific *dataset* between *group1* and *group2* and prints a table
    showing variable values in differing events, i.e., in events where the relative difference
    exceeds *epsilon*.
    """
    import numpy as np

    print("## Compare variable {} in dataset {} between {} and {}\n".format(
        variable, dataset, group1, group2))

    df1 = cache.get(dataset, group1)[["event", variable]]
    df2 = cache.get(dataset, group2)[["event", variable]]

    # get common events
    s1 = set(df1["event"].values)
    s2 = set(df2["event"].values)
    common_events = sorted(list(s1 & s2))
    print("### Stats\n")
    print("{}: {} events".format(group1, len(df1)))
    print("{}: {} events".format(group2, len(df2)))
    print("{} & {}: {} events".format(group1, group2, len(common_events)))

    # sort dataframes by event id column
    df1 = df1.sort_values(by=["event"])
    df2 = df2.sort_values(by=["event"])

    # get arrays (event,variable) for common events
    v1 = df1.values[np.isin(df1.values[:, 0], common_events)].astype(float)
    v2 = df2.values[np.isin(df2.values[:, 0], common_events)].astype(float)

    # verify that event ids are identical
    if not np.equal(v1[:, 0], v2[:, 0]).all():
        raise Exception("event ids are misaligned, please debug")

    # get variable difference
    diff = v1[:, 1] - v2[:, 1]
    reldiff = 2 * diff / (v1[:, 1] + v2[:, 1])

    # detect where relative differences exceed epsilon
    idxs = abs(reldiff) > epsilon

    if idxs.sum() == 0:
        print("\nno differences found")
        return

    print("\n### Differences in {} events\n".format(idxs.sum()))
    print("Variable: {}".format(variable))
    print("Sigma   : {:.6f}".format(diff.std()))

    headers = ["event", "{}".format(group1), "{}".format(group2), "{} - {}".format(group1, group2)]
    table = []
    for i, diverging in enumerate(idxs):
        if not diverging:
            continue
        table.append([df1["event"][i], df1[variable][i], df2[variable][i], diff[i]])

    print_table(table, headers=headers, floatfmt=".4f")


def visualize_variable(dataset=None, variables=None, epsilon=1e-5):
    """
    Creates a visualization for a specific *dataset* and *variables* and saves it in the plot
    directory. When *dataset* is *None*, all available datasets are used. When *variables* is
    *None*, the variables defined in the configuration are used.
    """
    import numpy as np

    # default datasets
    datasets = get_datasets(dataset)

    # default variables
    variables = get_variables(variables)

    def visualize(dataset, variable):
        diffs = {}
        for group1, group2 in itertools.combinations(config.get_groups(dataset), 2):
            # get common events
            df1 = cache.get(dataset, group1)[["event", variable]].sort_values(by=["event"])
            df2 = cache.get(dataset, group2)[["event", variable]].sort_values(by=["event"])
            s1 = set(df1["event"].values)
            s2 = set(df2["event"].values)
            common_events = sorted(list(s1 & s2))

            # get arrays (event,variable) for common events
            v1 = df1.values[np.isin(df1.values[:, 0], common_events)].astype(float)
            v2 = df2.values[np.isin(df2.values[:, 0], common_events)].astype(float)

            # get variable difference
            diff = v1[:, 1] - v2[:, 1]
            reldiff = 2 * diff / (v1[:, 1] + v2[:, 1])

            # store relative difference
            diffs[(group1, group2)] = reldiff

        # draw the comparison
        path = os.path.join(cli.plot_dir, "comparison__{}__{}.png".format(dataset, variable))
        draw_variable_comparison(dataset, variable, diffs, path, epsilon=epsilon)

    # loop over all datasets and variables
    for dataset in datasets:
        # skip when less than two groups are participating
        n_groups = len(config.get_groups(dataset))
        if n_groups < 2:
            print("only {} group(s) are synchronizing dataset {}, skip".format(n_groups, dataset))
            continue
        for variable in variables:
            visualize(dataset, variable)


def write_all():
    """
    Writes all tables and plots defined in the synchronization tools.
    """
    datasets = get_datasets()

    # write yields for all datasets
    for dataset in datasets:
        write_yields(dataset)

    # write all event comparison tables
    for dataset in datasets:
        write_event(dataset)

    # create some visualizations
    visualize_variable(variables=[
        "rho", "pu", "n_jets", "n_btags", "jet1_pt", "jet1_eta", "tau1_pt", "tau1_eta",
    ])
