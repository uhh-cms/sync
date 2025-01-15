# coding: utf-8

"""
Syncrhonization tools.
"""

from __future__ import annotations

__all__ = ["Tools"]

import os
import math
import argparse
import itertools
import subprocess
# import sys
# import copy
# import random
# import contextlib
# import collections

import numpy as np
import tabulate  # type: ignore[import-untyped]

from sync.config import Config
from sync.loader import DataLoader
from sync.utils import colored
from sync._types import Callable, Any


_expose_counter = 0


def expose(func: Callable) -> Callable:
    global _expose_counter
    func._exposed = _expose_counter  # type: ignore[attr-defined]
    _expose_counter += 1
    return func


def is_exposed(func: Callable) -> bool:
    return isinstance(getattr(func, "_exposed", None), int)


class Tools(object):

    missing_value = -999.0

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
        methods = [
            (name, getattr(self, name))
            for name in dir(self)
            if is_exposed(getattr(self, name))
        ]
        # sort
        methods.sort(key=lambda x: x[1]._exposed)
        # convert to dict
        return dict(methods)

    def _transpose(
        self,
        header: list[str],
        rows: list[Any],
    ) -> tuple[list[str], list[list[Any]]]:
        all_rows = [header] + rows

        # dimensions of the transposed table
        num_cols = len(all_rows)
        num_rows = len(all_rows[0])

        # transpose the columns
        new_rows = []
        for row_idx in range(num_rows):
            row = []
            for col_idx in range(num_cols):
                row.append(all_rows[col_idx][row_idx])
            new_rows.append(row)

        header = new_rows.pop(0)

        return header, new_rows

    def _print_table(self, *args, **kwargs) -> None:
        kwargs.setdefault("tablefmt", self.args.table_format)
        kwargs.setdefault("intfmt", "_")
        kwargs.setdefault("floatfmt", ".6f")

        if kwargs.pop("transpose", False):
            header = kwargs.get("headers")
            if not header:
                raise RuntimeError("cannot transpose table without headers")
            header, rows = self._transpose(header, args[0])
            # update old header and rows
            args = (rows,)
            kwargs["headers"] = header

        print(tabulate.tabulate(*args, **kwargs))

    def _print_header(self, level: int, msg: str) -> None:
        text = f"{(level + 1) * '#'} {msg}"
        print(f"\n{colored(text, **self.header_styles[level])}\n")

    def _show_plot(self, path: str) -> None:
        if not self.args.view_cmd:
            return

        subprocess.run(f"{self.args.view_cmd} '{path}'", shell=True, executable="/bin/bash")

    def _get_missing_value(self, dataset: str, group: str) -> float:
        missing = self.config.get_missing_value(dataset, group)
        return missing if missing is not None else self.missing_value

    def _inject_minimal_variables(
        self,
        variables: list[str],
        minimal: tuple[str, ...] = ("event", "run", "lumi"),
    ) -> None:
        # always add event, run and lumi at the beginning
        for v in minimal[::-1]:
            if v in variables:
                variables.remove(v)
            variables.insert(0, v)

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
        for cat in self.config.get_categories().values():
            print(f"    - {cat['name']}: {cat['expression']}")

        # variables
        print("\nvariables:")
        for name in self.config.get_variables():
            print(f"    - {name}")

    @expose
    def show_yields(self, dataset: str | None = None, transpose: bool = False) -> None:
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
            for cat in self.config.get_categories().values():
                line = [cat["name"]]
                for group in groups:
                    df = self.loader(dataset, group)
                    try:
                        line.append(sum(df.eval(cat["expression"])))
                    except ImportError:
                        raise
                    except Exception as e:
                        e.args = (
                            f"evaluation failed for group {group}, dataset {dataset}, category "
                            f"{cat['name']}: {e}",
                            *e.args[1:],
                        )
                        raise
                table.append(line)
            self._print_table(table, headers=headers, transpose=transpose)

        for dataset in datasets:
            show(dataset)
            print("")

    @expose
    def compare_yields(self, dataset: str, group1: str, group2: str, *, transpose: bool = False) -> None:
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
        for cat in self.config.get_categories().values():
            _df1 = df1[df1.eval(cat["expression"])]
            _df2 = df2[df2.eval(cat["expression"])]

            # get sets of the event id for simple comparison
            s1 = set(_df1["event"].values)
            s2 = set(_df2["event"].values)

            table.append((cat["name"], len(s1), len(s2), len(s1 & s2), len(s1 - s2), len(s2 - s1)))

        self._print_table(table, headers=headers, transpose=transpose)

    @expose
    def check_missing_events(
        self,
        dataset: str,
        group1: str,
        group2: str,
        variables: str | None = None,
        interactive: bool = True,
        transpose: bool = False,
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
        self._inject_minimal_variables(_variables)

        # helper to get the variable type
        vtype = lambda v: self.config.get_variables()[v]["type"]

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
                row.extend(vtype(v)(df[idxs][v].values[0]) for v in _variables)

                self._print_table([row], headers=headers, transpose=transpose)

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
        groups: str | list[str] | None = None,
        variables: str | None = None,
        interactive: bool = True,
        transpose: bool = False,
    ) -> None:
        """
        Traverses events in a *dataset* that are common to all *groups* and prints a table with
        specific *variables* per event. When *groups* is *None*, all participating groups are
        selected. When *variables* is *None*, the variables defined in the configuration are used.
        In case of multiple events, a prompt allows to either stop or continue the comparison when
        *interactive* is *True*.
        """
        # select groups and variables
        _groups = (
            self.config.get_groups(dataset)
            if groups is None
            else self.config.select_groups(groups)
        )
        _variables = self.config.select_variables(variables)
        self._inject_minimal_variables(_variables)

        # helper to get the variable type
        vtype = lambda v: self.config.get_variables()[v]["type"]

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
            headers = [] + _variables
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

                row = [group] + [vtype(v)(df[idxs][v].values[0]) for v in _variables]
                table.append(row)

            self._print_table(table, headers=headers, transpose=transpose)

            if i < len(common) - 1:
                print("")
                if interactive:
                    inp = input("press enter to continue, or any other key to stop: ").strip()
                    if inp:
                        break
                    print("")

    @expose
    def compare_event(
        self,
        dataset: str,
        event: int | None = None,
        variables: str | None = None,
        interactive: bool = True,
        transpose: bool = False,
    ) -> None:
        """
        Compares *variables* of an *event* given by its id in a specific *dataset*. When *variables*
        is None*, the variables defined in the configuration are used. When *event* is *None*, all
        events in that dataset compared. In case of multiple events, a prompt allows to either stop
        or continue the comparison when *interactive* is *True*.
        """
        # select groups and variables
        _groups = self.config.get_groups(dataset)
        _variables = self.config.select_variables(variables)
        self._inject_minimal_variables(_variables)

        # helper to get the variable type
        vtype = lambda v: self.config.get_variables()[v]["type"]

        # default events
        if not event:
            _events = list(set.union(*(set(self.loader(dataset, g)["event"].values) for g in _groups)))
        elif isinstance(event, list):
            _events = list(event)
        else:
            _events = [event]

        def compare(event: int) -> None:
            self._print_header(1, f"Comparison of event {event} in dataset {dataset}")

            selection = f"(event == {event})"
            table = [[group] for group in _groups]

            for group, row in zip(_groups, table):
                missing_value = self._get_missing_value(dataset, group)
                df = self.loader(dataset, group)
                idxs = df.eval(selection)
                n = sum(idxs)
                if n == 0:
                    # fill some missing value
                    row.extend(missing_value for _ in _variables)  # type: ignore[misc]
                else:
                    if n != 1:
                        raise Exception(
                            f"event {event} contained {n} times in dataset {dataset} of group "
                            f"{group}",
                        )
                    row.extend([vtype(v)(df[idxs][v].values[0]) for v in _variables])

            self._print_table(table, headers=["group"] + _variables, transpose=transpose)

        # loop over events
        for i, e in enumerate(_events):
            compare(e)
            if i < len(_events) - 1:
                print("")
                if interactive:
                    next_event = _events[i + 1]
                    inp = input(
                        f"press enter to continue with next event ({next_event}) or any other key "
                        "to stop: ",
                    ).strip()
                    if inp:
                        break

    @expose
    def compare_variable(
        self,
        dataset: str,
        variable: str,
        group1: str,
        group2: str,
        category: str | None = None,
        n_events: int = 10,
        epsilon: float = 1e-5,
        transpose: bool = False,
    ) -> None:
        """
        Compares a *variable* in a specific *dataset* between *group1* and *group2* and prints a
        table showing variable values in differing events, i.e., in events where the relative
        difference exceeds *epsilon*. When *category* is given, only events from that category are
        compared. The number of differing events is limited by *n_events*. All events are shown when
        it is set to a non-positive value.
        """
        msg = f"Compare variable {variable} in dataset {dataset} between {group1} and {group2}"
        if category is not None:
            msg += f" in category {category}"
        self._print_header(1, msg)

        # load data into pandas dataframes
        df1 = self.loader(dataset, group1)
        df2 = self.loader(dataset, group2)

        # apply category filter
        if category is not None:
            expr = self.config.get_categories()[category]["expression"]
            df1 = df1[df1.eval(expr)]
            df2 = df2[df2.eval(expr)]

        # reduce variables
        df1 = df1[["event", "run", "lumi", variable]]
        df2 = df2[["event", "run", "lumi", variable]]

        # get common events in both pandas dataframes
        suffixes = ["_x", "_y"]
        variables = [f"{variable}{suffix}" for suffix in suffixes]
        common_events = df1.merge(df2, how="inner", on=["event", "run", "lumi"], suffixes=suffixes)

        self._print_header(2, "Stats")
        print(f"{group1}: {len(df1)} events")
        print(f"{group2}: {len(df2)} events")
        print(f"{group1} & {group2}: {len(common_events)} common events")

        # sort dataframes by event id column
        common_events.sort_values(by=["event"])

        # get variable difference and relative difference, std deviation
        common_events["difference"] = common_events[variables[0]] - common_events[variables[1]]
        reldiff = 2 * common_events["difference"] / (common_events[variables[0]] + common_events[variables[1]])
        std = common_events["difference"].std()

        # detect where relative differences exceed epsilon
        idxs = abs(reldiff) > epsilon

        if idxs.sum() == 0:
            print("\nno differences found")
            return

        self._print_header(2, f"Differences in {idxs.sum()} events")
        print(f"variable: {variable}")
        print(f"sigma   : {std:.6f}")

        headers = ["event", group1, group2, f"{group1} - {group2}"]
        table = []
        vtype = self.config.get_variables()[variable]["type"]
        for i, diverging in enumerate(idxs):
            if not diverging:
                continue
            table.append([
                int(common_events["event"][i]),
                vtype(common_events[variables[0]][i]),
                vtype(common_events[variables[1]][i]),
                vtype(common_events["difference"][i]),
            ])
            if n_events > 0 and len(table) >= n_events:
                break
        print("")
        if n_events > 0:
            print(f"showing first {n_events} of {len(idxs)} events with differences in variable")
        self._print_table(table, headers=headers, transpose=transpose)

    @expose
    def draw_variable(
        self,
        dataset: str | None = None,
        variable: str | None = None,
        ref_group: str | None = None,
        category: str | None = None,
        bins: int = 20,
        normalize: bool = False,
    ) -> None:
        """
        Creates a histogram with number of *bins* also including a ratio relative to *ref_group*.
        The plot is created specific for a *dataset* and *variable* and saves it in the plot
        directory. When *dataset* is *None*, all available datasets are used. When *variables* is
        *None*, the variables defined in the configuration are used. When *category* is given, only
        events from that category are compared.
        """
        _datasets = self.config.select_datasets(dataset)
        _variables = self.config.select_variables(variable)
        _groups = sorted(self.config.get_groups(dataset))

        # default reference group
        if ref_group is None:
            ref_group = _groups[0]

        def visualize(
            dataset: str,
            variable: str,
            bins: int | list[float] = bins,
        ) -> None:
            """
            Helper function to load the groups data and draw the comparison.
            """
            variable_data = {}

            # prepare category expression
            cat = None if category is None else self.config.get_categories()[category]

            # get data for all groups
            for _group in _groups:
                df = self.loader(dataset, _group)
                if cat:
                    df = df[df.eval(cat["expression"])]
                df = df[["event", variable]].sort_values(by=["event"])
                variable_data[_group] = df[variable].values.astype(float)

            # check if the variable is categorical
            categorical_values = self.config.get_variables()[variable].get("labels", None)

            # draw the comparison
            path = os.path.join(self.args.plot_dir, f"ratio__{dataset}__{variable}.png")
            # show it when possible
            draw_hist_with_ratio(
                dataset=dataset,
                variable=variable,
                ref_group=ref_group,
                data=variable_data,
                bins=bins,
                path=path,
                normalize=normalize,
                categorical_values=categorical_values,
                missing_values={
                    g: self._get_missing_value(dataset, g)
                    for g in {ref_group, *_groups}
                },
                category=cat["label"] if cat else None,
            )
            print(f"created plot at {path}")
            self._show_plot(path)

        # loop over all datasets and variables
        for dataset in _datasets:
            for variable in _variables:
                visualize(dataset, variable, bins=bins)

    @expose
    def draw_variance(
        self,
        dataset: str | None = None,
        variables: str | None = None,
        epsilon: float = 1e-5,
    ) -> None:
        """
        Creates a visualization of the variance for a specific *dataset* and *variables* and saves
        it in the plot directory. When *dataset* is *None*, all available datasets are used. When
        *variables* is *None*, the variables defined in the configuration are used.
        """
        # select datasets and variables
        _datasets = self.config.select_datasets(dataset)
        _variables = self.config.select_variables(variables)

        def visualize(dataset: str, variable: str) -> None:
            diffs = {}
            for group1, group2 in itertools.combinations(self.config.get_groups(dataset), 2):
                # get common events
                df1 = self.loader(dataset, group1)[["event", variable]].sort_values(by=["event"])
                df2 = self.loader(dataset, group2)[["event", variable]].sort_values(by=["event"])
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
            path = os.path.join(self.args.plot_dir, f"comparison__{dataset}__{variable}.png")
            draw_variable_comparison(
                dataset=dataset,
                variable=variable,
                diffs=diffs,
                path=path,
                epsilon=epsilon,
            )
            print(f"created plot at {path}")
            self._show_plot(path)

        # loop over all datasets and variables
        for dataset in _datasets:
            # skip when less than two groups are participating
            n_groups = len(self.config.get_groups(dataset))
            if n_groups < 2:
                print(f"only {n_groups} group(s) are synchronizing dataset {dataset}, skip")
                continue
            for variable in _variables:
                visualize(dataset, variable)


def draw_hist_with_ratio(
    *,
    dataset: str,
    variable: str,
    ref_group: str,
    data: dict[str, np.ndarray],
    bins: int | list[float],
    path: str,
    normalize: bool,
    categorical_values: dict[int, str] | None,
    missing_values: dict[str, float],
    category: str | None = None,
) -> None:
    import mplhep as hep  # type: ignore[import-untyped]
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.style.use(hep.style.CMS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # categorical axis
    x_range = None
    if categorical_values is not None:
        bins = len(categorical_values)
        vals = list(categorical_values.keys())
        x_range = (min(vals) - 0.5, max(vals) + 0.5)

    fig, ax = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0))

    # plot reference group
    ref_data = data.pop(ref_group)
    ref_count, ref_edges, *_ = ax[0].hist(
        ref_data[ref_data != missing_values[ref_group]],
        bins=bins,
        range=x_range,
        label=ref_group,
        histtype="step",
        color=colors[0],
        density=normalize,
        linewidth=2,
    )
    ref_centers = (ref_edges[:-1] + ref_edges[1:]) / 2
    ax[1].hlines(1.0, ref_edges[0], ref_edges[-1], linestyle="-", color=colors[0])

    # get raw counts for ratio calculation when normalizing
    if normalize:
        ref_count, _ = np.histogram(ref_data[ref_data != missing_values[ref_group]], bins=bins)

    # plot rest of the groups
    for group, color in zip(data.keys(), colors[1:]):
        count, *_ = ax[0].hist(
            data[group],
            bins=ref_edges,
            label=group,
            histtype="step",
            color=color,
            density=normalize,
            linewidth=2,
        )
        # ratio plot relative to main group
        if normalize:
            count, _ = np.histogram(data[group], bins=ref_edges)
        ratio = count / ref_count
        ratio_err = ratio * (1 / ref_count + 1 / count)**0.5
        ax[1].errorbar(
            ref_centers,
            ratio,
            yerr=ratio_err,
            label=group,
            linestyle="",
            marker="o",
            color=color,
        )

    # styling and labels
    hep.cms.label("Private work", ax=ax[0], data=False, com=13.6)
    handles, labels = ax[0].get_legend_handles_labels()
    y_label_x_offset = -0.14 if normalize else -0.09

    handles = [
        Line2D([0], [0], color=polygon.get_edgecolor(), linestyle="-", linewidth=3)
        for polygon in handles
    ]
    ax[0].legend(handles=handles, labels=labels, loc="upper right", title=f"Dataset: {dataset}")
    ax[0].set_ylabel("Normalized entries" if normalize else "Entries")
    ax[0].yaxis.set_label_coords(y_label_x_offset, 1.0)
    ax[1].set_ylim(0.2, 1.8)
    ax[1].set_xlabel(variable)
    ax[1].set_ylabel(f"1 / {ref_group}", loc="center")
    ax[1].yaxis.set_label_coords(y_label_x_offset, 0.5)
    ax[1].grid(axis="y", linestyle="-", linewidth=1)
    # set tick labels when axis is categorical
    if categorical_values:
        ax[0].set_xticks(list(categorical_values.keys()))
        ax[1].set_xticks(list(categorical_values.keys()))
        ax[1].set_xticklabels(list(categorical_values.values()))

    # category label
    if category:
        ax[0].annotate(category, xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top")

    fig.savefig(path, dpi=120, bbox_inches="tight")


def draw_variable_comparison(
    *,
    dataset: str,
    variable: str,
    diffs: dict[tuple[str, str], np.ndarray],
    path: str,
    epsilon: float = 1e-5,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx  # type: ignore[import-untyped]

    # plot title
    title = f"Dataset '{dataset}', Variable '{variable}'"

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
        return i * 2 * math.pi / len(groups) + math.pi / 2.0

    # helper to translate from polar to cartesian coordinates
    def p2c(r, phi, y_offset=0.0):
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
    nx.draw_networkx_labels(graph, {0: p2c(1.1, math.pi / 2.0)}, {0: title}, font_size=18)

    # add edges, based on differences
    for (group1, group2), _diffs in diffs.items():
        # the comparison is only deemed successful if all differences are below epsilon
        color = "green" if (abs(_diffs) <= epsilon).all() else "red"

        # compute a weight for the edge between 1 and 10, based on a projection into a single value
        diff = abs(_diffs).mean()
        min_width, max_width, scale = 2, 10, 0.75
        weight = (
            min_width
            if diff <= epsilon
            else (min_width + (max_width - min_width) * math.tanh(diff * scale))
        )

        nx.draw_networkx_edges(
            graph,
            node_pos,
            [(groups.index(group1), groups.index(group2))],
            width=weight,
            edge_color=color,
        )

    # save it
    print(f"save comparison plot of variable {variable} in dataset {dataset} at {path}")
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
