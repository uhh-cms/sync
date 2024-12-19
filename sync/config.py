# coding: utf-8

"""
Simple helpers to access json or yaml configurations.
"""

from __future__ import annotations

__all__ = ["Config"]

import os
import pathlib
import itertools
import fnmatch

import yaml

from sync.utils import DotDict


class Config(DotDict):

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> Config:
        path = os.path.expandvars(os.path.expanduser(str(path)))
        with open(path, "rb") as f:
            return cls(**yaml.safe_load(f))  # type: ignore[return-value]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # dataset names and group names should not contain spaces
        for dataset in self.get_datasets():
            if " " in dataset:
                raise ValueError(f"datasets should not contain spaces: '{dataset}'")
        for group in self.get_groups():
            if " " in group:
                raise ValueError(f"groups should not contain spaces: '{group}'")

    def get_globals(self) -> list[str]:
        return [
            *self.get_datasets(),
            *self.get_groups(),
            *self.get_variables(),
        ]

    def get_datasets(self, group: str | None = None) -> list[str]:
        if group is None:
            return list(self["datasets"].keys())

        return [
            dataset for dataset, data in self["datasets"].items()
            if group in data["groups"]
        ]

    def select_datasets(self, dataset: str | list[str] | None = None) -> list[str]:
        all_datasets = self.get_datasets()

        if dataset is None:
            return all_datasets

        selected_datasets: list[str] = []
        for pattern in (dataset if isinstance(dataset, list) else [dataset]):
            for d in all_datasets:
                if d not in selected_datasets and fnmatch.fnmatch(d, pattern):
                    selected_datasets.append(d)

        self.raise_no_match(selected_datasets, dataset, "datasets")
        return selected_datasets

    def get_groups(self, dataset: str | None = None) -> list[str]:
        if dataset is None:
            return list(set(sum((self.get_groups(d) for d in self.get_datasets()), [])))

        return list(self["datasets"][dataset]["groups"].keys())

    def select_groups(self, group: str | list[str] | None = None) -> list[str]:
        all_groups = self.get_groups()

        if group is None:
            return all_groups

        selected_groups: list[str] = []
        for pattern in (group if isinstance(group, list) else [group]):
            for d in all_groups:
                if d not in selected_groups and fnmatch.fnmatch(d, pattern):
                    selected_groups.append(d)

        self.raise_no_match(selected_groups, group, "groups")
        return selected_groups

    def raise_no_match(
        self,
        selected_var: list[str],
        var: str | list[str] | None,
        var_str: str,
    ) -> None:
        if not selected_var:
            raise ValueError(f"no {var_str} matched given '{var}'")

    def get_files(self, dataset: str, group: str) -> dict[str | int, str]:
        files = self["datasets"][dataset]["groups"][group]["files"]
        if isinstance(files, dict):
            return files
        if isinstance(files, list):
            return dict(enumerate(files))
        raise TypeError(f"field 'files' must be a mapping or sequence, got '{files}'")

    def get_transformation(self, dataset: str, group: str) -> str | None:
        data = self["datasets"][dataset]["groups"][group]
        return data.get("transform") if isinstance(data, dict) else None

    def get_variables(self) -> list[str]:
        return self["variables"]

    def select_variables(self, variable: str | list[str] | None = None) -> list[str]:
        all_variables = self.get_variables()
        if variable is None:
            return all_variables

        selected_variables: list[str] = []
        for pattern in (variable if isinstance(variable, list) else [variable]):
            for d in all_variables:
                if d not in selected_variables and fnmatch.fnmatch(d, pattern):
                    selected_variables.append(d)

        self.raise_no_match(selected_variables, variable, "variables")
        return selected_variables

    def get_categories(self) -> dict[str, str]:
        return self["categories"]

    def _dataset_group_valid(self, dataset: str, group: str) -> bool:
        return dataset in self["datasets"] and group in self["datasets"][dataset]["groups"]

    def get_dataset_group_combinations(
        self,
        dataset: str | list[str] | None = None,
        group: str | list[str] | None = None,
    ) -> list[tuple[str, str]]:
        # sanitize datasets
        if dataset is None:
            datasets = self.get_datasets()
        elif not isinstance(dataset, list):
            datasets = [dataset]
        else:
            datasets = list(dataset)

        # sanitize groups
        if group is None:
            groups = self.get_groups()
        elif not isinstance(group, list):
            groups = [group]
        else:
            groups = list(group)

        # return only valid combinations
        return [
            (dataset, group) for dataset, group in itertools.product(datasets, groups)
            if self._dataset_group_valid(dataset, group)
        ]
