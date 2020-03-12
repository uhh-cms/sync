# coding: utf-8

"""
Simple helpers to access json or yaml configurations.
"""


__all__ = ["Config"]


import itertools


class ForwardDict(object):

    def __init__(self, *args, **kwargs):
        super(ForwardDict, self).__init__()

        self._data = dict(*args, **kwargs)

    @property
    def data(self):
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value):
        self._data[item] = value

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def update(self, data):
        self._data.update(data)

    def clear(self):
        self._data.clear()


class Config(ForwardDict):

    @classmethod
    def load(cls, path):
        import oyaml
        with open(path, "rb") as f:
            return cls(oyaml.load(f, Loader=oyaml.SafeLoader))

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

        # dataset names and group names should not contain spaces
        for dataset in self.get_datasets():
            if " " in dataset:
                raise ValueError("datasets should not contain spaces: '{}'".format(dataset))
        for group in self.get_groups():
            if " " in group:
                raise ValueError("groups should not contain spaces: '{}'".format(group))

    def get_globals(self):
        g = []

        # dataset names
        g.extend(self.get_datasets())

        # group names
        g.extend(self.get_groups())

        return g

    def get_datasets(self, group=None):
        if group:
            return [
                dataset for dataset, data in self["datasets"].items()
                if group in data["groups"]
            ]
        else:
            return list(self["datasets"].keys())

    def get_groups(self, dataset=None):
        if dataset:
            return list(self["datasets"][dataset]["groups"].keys())
        else:
            return list(set(sum((self.get_groups(d) for d in self.get_datasets()), [])))

    def _dataset_group_valid(self, dataset, group):
        return dataset in self["datasets"] and group in self["datasets"][dataset]["groups"]

    def get_dataset_group_combinations(self, dataset=None, group=None):
        # sanitize datasets
        if not dataset:
            datasets = self.get_datasets()
        elif not isinstance(dataset, list):
            datasets = [dataset]
        else:
            datasets = list(dataset)

        # sanitize groups
        if not group:
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

    def get_file(self, dataset, group):
        data = self["datasets"][dataset]["groups"][group]
        return data["path"] if isinstance(data, dict) else data

    def get_translation(self, dataset, group):
        data = self["datasets"][dataset]["groups"][group]
        return data.get("translation") if isinstance(data, dict) else None

    def get_variables(self):
        return self["variables"]

    def get_categories(self):
        return self["categories"]
