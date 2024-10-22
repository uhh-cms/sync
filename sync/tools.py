# coding: utf-8

"""
Syncrhonization tools.
"""

from __future__ import annotations

__all__ = ["Tools"]

import argparse
# import os
# import sys
# import math
# import copy
# import random
# import contextlib
# import collections
# import itertools

import tabulate  # type: ignore[import-untyped]

from sync.config import Config
from sync.loader import DataLoader
from sync.utils import colored
from sync._types import Callable


def expose(func: Callable) -> Callable:
    func._exposed = True  # type: ignore[attr-defined]
    return func


def is_exposed(func: Callable) -> bool:
    return getattr(func, "_exposed", False)


class Tools(object):

    header_styles = {
        1: {"color": "cyan", "style": "bright"},
        2: {"color": "cyan"},
        3: {"color": "default"},
    }

    def __init__(self, args: argparse.Namespace, config: Config, loader: DataLoader) -> None:
        super().__init__()

        self.args = args
        self.config = config
        self.loader = loader

    def get_exposed_methods(self) -> dict[str, Callable]:
        return {
            name: getattr(self, name)
            for name in dir(self)
            if is_exposed(getattr(self, name))
        }

    def _print_table(self, *args, **kwargs) -> None:
        kwargs.setdefault("tablefmt", self.args.table_format)
        print(tabulate.tabulate(*args, **kwargs))

    def _print_header(self, level: int, msg: str) -> None:
        text = f"{(level + 1) * '#'} {msg}"
        print(f"\n{colored(text, **self.header_styles[level])}\n")

    @expose
    def print_config(self) -> None:
        """
        Prints a summary of the current configuration.
        """
        # datasets and groups
        print("datasets and participating groups:")
        for dataset in self.config.get_datasets():
            print(f"    dataset: {dataset}")
            for group in self.config.get_groups(dataset):
                print(f"        group: {group}")

        # categories
        print("\ncategories:")
        for name, expr in self.config.get_categories().items():
            print(f"    - {name}: {expr}")

        # variables
        print("\nvariables:")
        for name in self.config.get_variables():
            print(f"    - {name}")

    @expose
    def show_yields(self, dataset: str | None = None) -> None:
        """
        Shows the yields for all groups in a specific *dataset*. When *None*, all datases are
        evaluated sequentially.
        """
        datasets = self.config.select_datasets(dataset)

        def show(dataset: str) -> None:
            self._print_header(1, f"Yields for dataset {dataset}")
            groups = self.config.get_groups(dataset)
            headers = ["category / group"] + groups
            table = []
            for cat, cat_expr in self.config.get_categories().items():
                line = [cat]
                for group in groups:
                    df = self.loader(dataset, group)
                    try:
                        line.append(sum(df.eval(cat_expr)))
                    except ImportError:
                        raise
                    except Exception as e:
                        e.args = (
                            f"evaluation failed for group {group}, dataset {dataset}, category "
                            f"{cat}: {e}",
                            *e.args[1:],
                        )
                        raise
                table.append(line)
            self._print_table(table, headers=headers)

        for dataset in datasets:
            show(dataset)
            print("")

    @expose
    def compare_yields(self, dataset: str, group1: str, group2: str) -> None:
        """
        Compares the yields in a specific *dataset* between *group1* and *group2*, subdivided into
        all known categories.
        """
        self._print_header(1, f"Yield comparison for dataset {dataset} between {group1} and {group2}")

        headers = [
            "category / type",
            group1,
            group2,
            "common",
            f"{group1} - {group2}",
            f"{group2} - {group1}",
        ]
        table = []

        df1 = self.loader(dataset, group1)
        df2 = self.loader(dataset, group2)
        for cat, cat_expr in self.config.get_categories().items():
            _df1 = df1[df1.eval(cat_expr)]
            _df2 = df2[df2.eval(cat_expr)]

            # get sets of the event id for simple comparison
            s1 = set(_df1["event"].values)
            s2 = set(_df2["event"].values)

            table.append((cat, len(s1), len(s2), len(s1 & s2), len(s1 - s2), len(s2 - s1)))

        self._print_table(table, headers=headers)

    @expose
    def check_missing_events(
        self,
        dataset: str,
        group1: str,
        group2: str,
        variables: str | None = None,
        interactive: bool = True,
    ) -> None:
        """
        Traverses missing events between *group1* and *group2* in a specific *dataset* and prints a
        table with specific *variables* per event. When *variables* is *None*, all variables defined
        in the configuration are used. In case of multiple events, a prompt allows to either stop or
        continue the comparison when *interactive* is *True*.
        """
        self._print_header(1, f"Missing events for dataset {dataset} between {group1} and {group2}")

        # select variables
        _variables = self.config.select_variables(variables)

        # get data frames
        df1 = self.loader(dataset, group1)
        df2 = self.loader(dataset, group2)

        # create event set differences
        s1 = set(df1["event"].values)
        s2 = set(df2["event"].values)
        diff12 = s1 - s2
        diff21 = s2 - s1

        self._print_header(2, "Stats and differences")
        print(f"{group1}: {len(s1)} events")
        print(f"{group2}: {len(s2)} events")
        print(f"{group1} - {group2}: {len(diff12)} events")
        print(f"{group2} - {group1}: {len(diff21)} events")
        print(f"{group1} | {group2}: {len(s1 | s2)} events")
        print(f"{group1} & {group2}: {len(s1 & s2)} events")

        def traverse_diff(group1, group2, df, diff, can_reverse=False):
            self._print_header(2, f"Traversing {group1} - {group2}")

            print(f"missing: {','.join(str(e) for e in diff)}\n")

            for event in diff:
                headers = [f"# {event}"] + _variables
                row = [group1]

                idxs = df.eval(f"(event == {event})")
                if idxs.sum() != 1:
                    raise Exception(
                        f"event {event} contained {len(idxs)} times in dataset {dataset} of group"
                        f" {group1}",
                    )
                row.extend(df[idxs][v].values[0] for v in _variables)

                self._print_table([row], headers=headers, floatfmt=".4f")

                if interactive:
                    print("")
                    if can_reverse:
                        inp = input(
                            "press enter to continue, 'r' to reverse groups, or any other key to "
                            "stop: ",
                        ).strip()
                        if inp.lower() == "r":
                            return True
                        if inp:
                            return False
                    else:
                        inp = input("press enter to continue, or any other key to stop: ").strip()
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

    @expose
    def check_common_events(
        self,
        dataset: str,
        groups: str | None = None,
        variables: str | None = None,
        interactive: bool = True,
    ) -> None:
        """
        Traverses events in a *dataset* that are common to all *groups* and prints a table with
        specific *variables* per event. When *groups* is *None*, all participating groups are
        selected. When *variables* is *None*, the variables defined in the configuration are used.
        In case of multiple events, a prompt allows to either stop or continue the comparison when
        *interactive* is *True*.
        """
        # select groups and variables
        _groups = self.config.select_groups(groups)
        _variables = self.config.select_variables(variables)

        self._print_header(1, f"Common events for dataset {dataset} between {', '.join(_groups)}")

        # get common events
        common: set[int] = set()
        for group in _groups:
            df = self.loader(dataset, group)
            s = set(df["event"].values)
            common = s if not common else set.intersection(common, s)

        print(f"{' & '.join(_groups)}: {len(common)} common events\n")

        # traverse common events
        for i, event in enumerate(common):
            headers = [f"# {event}"] + _variables
            table = []

            selection = f"(event == {event})"
            for group in _groups:
                df = self.loader(dataset, group)
                idxs = df.eval(selection)
                if idxs.sum() != 1:
                    raise Exception(
                        f"event {event} contained {len(idxs)} times in dataset {dataset} of group "
                        f"{group}",
                    )
                row = [group] + [df[idxs][v].values[0] for v in _variables]
                table.append(row)

            self._print_table(table, headers=headers, floatfmt=".4f")

            if i < len(common) - 1:
                print("")
                if interactive:
                    inp = input("press enter to continue, or any other key to stop: ").strip()
                    if inp:
                        break
                    print("")

    # TODO
    # def compare_event(dataset, event=None, variables=None, interactive=True):
    #     """
    #     Compares *variables* of an *event* given by its id in a specific *dataset*. When *variables* is
    #     *None*, the variables defined in the configuration are used. When *event* is *None*, all events
    #     in that dataset compared. In case of multiple events, a prompt allows to either stop or continue
    #     the comparison when *interactive* is *True*.
    #     """
    #     # get participating groups
    #     groups = config.get_groups(dataset)

    #     # default variables
    #     variables = get_variables(variables)

    #     # default events
    #     if not event:
    #         events = list(set.union(*(set(cache.get(dataset, g)["event"].values) for g in groups)))
    #     elif isinstance(event, list):
    #         events = list(event)
    #     else:
    #         events = [event]

    #     def compare(event):
    #         print("\n## Comparison of event {} in dataset {}\n".format(event, dataset))

    #         selection = "(event == {})".format(event)
    #         table = [[group] for group in groups]

    #         for group, row in zip(groups, table):
    #             df = cache.get(dataset, group)
    #             idxs = df.eval(selection)
    #             n = sum(idxs)
    #             if n == 0:
    #                 # fill some missing value
    #                 row.extend(MISSING for _ in variables)
    #             else:
    #                 if n != 1:
    #                     raise Exception("event {} contained {} times in dataset {} of group {}".format(
    #                         event, n, dataset, group))
    #                 for v in variables:
    #                     val = df[idxs][v].values[0]
    #                     if isinstance(val, int) or val == -1:
    #                         val = str(int(val))  # numpy-safe conversion
    #                     row.append(val)

    #         print_table(table, headers=["group"] + list(variables), floatfmt=".4f")

    #     # loop over events
    #     for i, event in enumerate(events):
    #         compare(event)
    #         if i < len(events) - 1:
    #             print("")
    #             if interactive:
    #                 next_event = events[i + 1]
    #                 inp = raw_input("press enter to continue with next event ({}) or any other key to "
    #                     "stop: ".format(next_event)).strip()
    #                 if inp:
    #                     break

    # def compare_variable(dataset, variable, group1, group2, epsilon=1e-5):
    #     """
    #     Compares a *variable* in a specific *dataset* between *group1* and *group2* and prints a table
    #     showing variable values in differing events, i.e., in events where the relative difference
    #     exceeds *epsilon*.
    #     """
    #     import numpy as np

    #     print("## Compare variable {} in dataset {} between {} and {}\n".format(
    #         variable, dataset, group1, group2))

    #     df1 = cache.get(dataset, group1)[["event", variable]]
    #     df2 = cache.get(dataset, group2)[["event", variable]]

    #     # get common events
    #     s1 = set(df1["event"].values)
    #     s2 = set(df2["event"].values)
    #     common_events = sorted(list(s1 & s2))
    #     print("### Stats\n")
    #     print("{}: {} events".format(group1, len(df1)))
    #     print("{}: {} events".format(group2, len(df2)))
    #     print("{} & {}: {} events".format(group1, group2, len(common_events)))

    #     # sort dataframes by event id column
    #     df1 = df1.sort_values(by=["event"])
    #     df2 = df2.sort_values(by=["event"])

    #     # get arrays (event,variable) for common events
    #     v1 = df1.values[np.isin(df1.values[:, 0], common_events)].astype(float)
    #     v2 = df2.values[np.isin(df2.values[:, 0], common_events)].astype(float)

    #     # verify that event ids are identical
    #     if not np.equal(v1[:, 0], v2[:, 0]).all():
    #         raise Exception("event ids are misaligned, please debug")

    #     # get variable difference
    #     diff = v1[:, 1] - v2[:, 1]
    #     reldiff = 2 * diff / (v1[:, 1] + v2[:, 1])

    #     # detect where relative differences exceed epsilon
    #     idxs = abs(reldiff) > epsilon

    #     if idxs.sum() == 0:
    #         print("\nno differences found")
    #         return

    #     print("\n### Differences in {} events\n".format(idxs.sum()))
    #     print("Variable: {}".format(variable))
    #     print("Sigma   : {:.6f}".format(diff.std()))

    #     headers = ["event", "{}".format(group1), "{}".format(group2), "{} - {}".format(group1, group2)]
    #     table = []
    #     for i, diverging in enumerate(idxs):
    #         if not diverging:
    #             continue
    #         table.append([df1["event"][i], df1[variable][i], df2[variable][i], diff[i]])

    #     print_table(table, headers=headers, floatfmt=".4f")

    # TODO
    # def visualize_variable(dataset=None, variables=None, epsilon=1e-5):
    #     """
    #     Creates a visualization for a specific *dataset* and *variables* and saves it in the plot
    #     directory. When *dataset* is *None*, all available datasets are used. When *variables* is
    #     *None*, the variables defined in the configuration are used.
    #     """
    #     import numpy as np

    #     # default datasets
    #     datasets = get_datasets(dataset)

    #     # default variables
    #     variables = get_variables(variables)

    #     def visualize(dataset, variable):
    #         diffs = {}
    #         for group1, group2 in itertools.combinations(config.get_groups(dataset), 2):
    #             # get common events
    #             df1 = cache.get(dataset, group1)[["event", variable]].sort_values(by=["event"])
    #             df2 = cache.get(dataset, group2)[["event", variable]].sort_values(by=["event"])
    #             s1 = set(df1["event"].values)
    #             s2 = set(df2["event"].values)
    #             common_events = sorted(list(s1 & s2))

    #             # get arrays (event,variable) for common events
    #             v1 = df1.values[np.isin(df1.values[:, 0], common_events)].astype(float)
    #             v2 = df2.values[np.isin(df2.values[:, 0], common_events)].astype(float)

    #             # get variable difference
    #             diff = v1[:, 1] - v2[:, 1]
    #             reldiff = 2 * diff / (v1[:, 1] + v2[:, 1])

    #             # store relative difference
    #             diffs[(group1, group2)] = reldiff

    #         # draw the comparison
    #         path = os.path.join(cli.plot_dir, "comparison__{}__{}.png".format(dataset, variable))
    #         draw_variable_comparison(dataset, variable, diffs, path, epsilon=epsilon)

    #     # loop over all datasets and variables
    #     for dataset in datasets:
    #         # skip when less than two groups are participating
    #         n_groups = len(config.get_groups(dataset))
    #         if n_groups < 2:
    #             print("only {} group(s) are synchronizing dataset {}, skip".format(n_groups, dataset))
    #             continue
    #         for variable in variables:
    #             visualize(dataset, variable)


####################################################################################################
# old code that needs porting / refactoring for additiona sync tools


# def draw_variable_comparison(dataset, variable, diffs, path, epsilon=1e-5):
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import networkx as nx

#     # plot title
#     title = "Dataset '{}', Variable '{}'".format(dataset, variable)

#     # diffs is a dict with keys being 2-tuples of groups,
#     # extract sorted groups
#     groups = sorted(list(set(sum((list(tpl) for tpl in diffs.keys()), []))))

#     # create a networkx graph
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_xlim(-1.1, 1.1)
#     ax.set_ylim(-1.1, 1.2)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     fig.tight_layout()
#     graph = nx.Graph()

#     # helper to compute the rotation for group i
#     def i2phi(i):
#         # add 90 deg offset to start at the top
#         return i * 2 * math.pi / len(groups) + math.pi / 2.

#     # helper to translate from polar to cartesian coordinates
#     def p2c(r, phi, y_offset=0.):
#         return (r * math.cos(phi), r * math.sin(phi) + y_offset)

#     # prepare graph attributes
#     node_pos = {}
#     label_pos = {}
#     labels = {}
#     for i, group in enumerate(groups):
#         # y offset to align the shape vertically
#         y_offset = -0.5 * (1 + math.sin(i2phi(len(groups) // 2)))
#         # calculate positions
#         phi = i2phi(i)
#         node_pos[i] = p2c(0.8, phi, y_offset)
#         label_pos[i] = p2c(0.95, phi, y_offset)
#         labels[i] = group
#         graph.add_node(i, pos=node_pos[i])

#     # draw nodes and labels
#     nx.draw(graph, node_pos, node_color="blue")
#     nx.draw_networkx_labels(graph, label_pos, labels, font_size=14)
#     nx.draw_networkx_labels(graph, {0: p2c(1.1, math.pi / 2.)}, {0: title}, font_size=18)

#     # add edges, based on differences
#     for (group1, group2), _diffs in diffs.items():
#         # compile the differences into a single value
#         # method 1: take the absolute variance of non-zero values
#         idxs = _diffs != 0
#         diff = abs(_diffs[_diffs != 0].var()) if idxs.any() else 0.
#         # method 2: take the std of non-zero values
#         # diff = _diffs[_diffs != 0].std() if idxs.any() else 0.

#         # define a line weight between 1 and 10 by using an activation-like approach
#         # see https://www.wolframalpha.com/input/?i=plot+10+*+tanh%28x+*+2%29+for+x%3D0+to+1.
#         weight = 2 if diff <= epsilon else max(1, 10 * math.tanh(diff * 2))

#         # define a line color
#         color = "green" if diff <= epsilon else "red"

#         nx.draw_networkx_edges(graph, node_pos, [(groups.index(group1), groups.index(group2))],
#             width=weight, edge_color=color)

#     # save it
#     print("save comparison plot of variable {} in dataset {} at {}".format(variable, dataset, path))
#     if os.path.exists(path):
#         os.remove(path)
#     fig.savefig(path, dpi=120, bbox_inches="tight")
#     plt.close(fig)


# def write_event(dataset, event=None, variables=None):
#     """
#     Writes the event comparison tables obtained by :py:func:`compare_event` into a file for a
#     specific *dataset* and *variables* for a selected *event*. When *None*, the *test_events* list
#     in the configuration entry for that dataset is used. *variables* is forwarded to
#     :py:func:`compare_event`.
#     """
#     # default events
#     if not event:
#         events = config["datasets"][dataset]["test_events"]

#     path = os.path.join(cli.table_dir, "events__{}.md".format(dataset))
#     if os.path.exists(path):
#         os.remove(path)

#     print("write event comparison for dataset {} to {}".format(dataset, path))

#     with open(path, "w") as f:
#         with change_stdout(f):
#             compare_event(dataset, events, variables=variables, interactive=False)


# def write_yields(dataset=None):
#     """
#     Writes the yield tables obtained by :py:func:`show_yields` into a file per dataset in the
#     table directory. When *dataset* is *None*, all datasets are evaluated sequentially.
#     """
#     datasets = get_datasets(dataset)

#     for dataset in datasets:
#         path = os.path.join(cli.table_dir, "yields__{}.md".format(dataset))
#         if os.path.exists(path):
#             os.remove(path)

#         print("write yields for dataset {} to {}".format(dataset, path))

#         with open(path, "w") as f:
#             with change_stdout(f):
#                 show_yields(dataset)


# def write_all():
#     """
#     Writes all tables and plots defined in the synchronization tools.
#     """
#     datasets = get_datasets()

#     # write yields for all datasets
#     for dataset in datasets:
#         write_yields(dataset)

#     # write all event comparison tables
#     for dataset in datasets:
#         write_event(dataset)

#     # create some visualizations
#     visualize_variable(variables=[
#         "rho", "pu", "n_jets", "n_btags", "jet1_pt", "jet1_eta", "tau1_pt", "tau1_eta",
#     ])
