# coding: utf-8

"""
Data loader and cache.
"""

from __future__ import annotations

__all__ = ["DataLoader"]

import os
import shutil
import urllib
import getpass
import subprocess
import collections

import pandas as pd

from sync.config import Config
from sync import transformations


class DataLoader(object):

    # template for fetching files through scp over lxplus
    SCP_TMPL = (
        "LC_ALL=C"
        " scp "
        " -o StrictHostKeyChecking=no"
        " -o IdentitiesOnly=yes"
        " -o PubkeyAuthentication=no"
        " -o GSSAPIDelegateCredentials=yes"
        " -o GSSAPITrustDNS=yes"
        " -o GSSAPIAuthentication=yes"
        " -o UserKnownHostsFile=/dev/null"
        " -o LogLevel=ERROR"
        " {user}@lxplus.cern.ch:{remote_path} {local_path}"
    )

    def __init__(self, config: Config, cache_dir: str) -> None:
        super().__init__()

        self.config = config
        self.cache_dir = cache_dir
        self.data: dict[str, dict[str, pd.DataFrame]] = collections.defaultdict(dict)

    def __call__(self, dataset, group):
        return self.load(dataset, group)

    def get_cache_path(
        self,
        dataset: str,
        group: str,
        file_key: int | str,
        /,
        ext: str = "csv",
    ) -> str:
        return os.path.join(self.cache_dir, f"{group}__{dataset}__{file_key}.{ext}")

    def flush(self, dataset: str | None = None, group: str | None = None) -> None:
        for dataset, group in self.config.get_dataset_group_combinations(dataset, group):
            for file_key, file_path in self.config.get_files(dataset, group).items():
                ext = os.path.splitext(file_path)[1][1:]
                cache_path = self.get_cache_path(dataset, group, file_key, ext)
                if not os.path.exists(cache_path):
                    continue
                print(f"remove cached file '{file_key}' for dataset {dataset} and group {group}")
                os.remove(cache_path)

    def has(self, dataset: str, group: str) -> bool:
        return dataset in self.data and group in self.data[dataset]

    def load(self, dataset: str, group: str) -> pd.DataFrame:
        if not self.has(dataset, group):
            # make sure all files are loaded
            self.fetch(dataset, group)

            # load contents as dataframes
            dfs = {
                file_key: self._load_impl(dataset, group, file_key, file_path)
                for file_key, file_path in self.config.get_files(dataset, group).items()
            }

            # apply transformations
            dfs = {
                file_key: self._transform(dataset, group, file_key, df)
                for file_key, df in dfs.items()
            }

            # concat and save the dataframe
            self.data[dataset][group] = pd.concat(list(dfs.values()))
            del dfs

        return self.data[dataset][group]

    def _load_impl(
        self,
        dataset: str,
        group: str,
        file_key: int | str,
        file_path: str,
    ) -> pd.DataFrame:
        cache_path = self._fetch_impl(dataset, group, file_key, file_path)

        # load csv via pandas
        return pd.read_csv(cache_path)

    def fetch(self, dataset: str | None = None, group: str | None = None) -> None:
        # prepare combinations of groups and datasets, then loop through all files
        for dataset, group in self.config.get_dataset_group_combinations(dataset, group):
            for file_key, file_path in self.config.get_files(dataset, group).items():
                self._fetch_impl(dataset, group, file_key, file_path)

    def _fetch_impl(self, dataset: str, group: str, file_key: int | str, file_path: str, /) -> str:
        ext = os.path.splitext(file_path)[1][1:]
        cache_path = self.get_cache_path(dataset, group, file_key, ext)

        # do nothing when already loaded
        if os.path.exists(cache_path):
            return cache_path

        # download or copy the data to the data dir
        if file_path.startswith("http://") or file_path.startswith("https://"):
            print(f"download dataset {dataset} ({file_key}) for group {group} from {file_path}")
            urllib.request.urlretrieve(file_path, cache_path)

        elif os.path.exists(file_path):
            print(f"copy dataset {dataset} ({file_key}) for group {group} from {file_path}")
            shutil.copy2(file_path, cache_path)

        elif file_path.startswith(("/afs/", "/eos/")):
            cmd = self.SCP_TMPL.format(
                user=os.environ.get("KRB_USER", getpass.getuser()),
                remote_path=file_path,
                local_path=cache_path,
            )
            print(f"scp dataset {dataset} ({file_key}) for group {group} from {file_path}")
            subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)

        else:
            raise RuntimeError(f"no method implemented to fetch '{file_path}'")

        # change permissions
        os.chmod(cache_path, 0o0664)

        return cache_path

    def _transform(
        self,
        dataset: str,
        group: str,
        file_key: int | str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        # get the transformation
        transform_name = self.config.get_transformation(dataset, group)
        if not transform_name:
            return df

        # get the transformation function
        transform_func = getattr(transformations, f"transform_{transform_name}", None)
        if not callable(transform_func):
            raise RuntimeError(f"transformation '{transform_name}' not found or not callable")

        # invoke it
        return transform_func(dataset, group, file_key, df)
