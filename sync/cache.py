# coding: utf-8

"""
Cache object that downloads and holds files for synchronization.
"""


import os
import subprocess
import shutil
import getpass
import collections

from six.moves import urllib

from sync import transformations


class Cache(object):

    SCP_TMPL = "LC_ALL=C scp " \
        "-o StrictHostKeyChecking=no " \
        "-o IdentitiesOnly=yes " \
        "-o PubkeyAuthentication=no " \
        "-o GSSAPIDelegateCredentials=yes " \
        "-o GSSAPITrustDNS=yes " \
        "-o GSSAPIAuthentication=yes " \
        "-o UserKnownHostsFile=/dev/null " \
        "-o LogLevel=ERROR " \
        "{user}@lxplus.cern.ch:{remote_path} {local_path}"

    def __init__(self, cli, config):
        super(Cache, self).__init__()

        self.cli = cli
        self.config = config
        self.data = collections.defaultdict(dict)

    def __call__(self, dataset, group):
        return self.get(dataset, group)

    def get_cache_path(self, dataset, group, file_key, ext="root"):
        return os.path.join(self.cli.cache_dir, "{}__{}__{}.{}".format(
            dataset, file_key, group, ext))

    def has(self, dataset, group):
        return dataset in self.data and group in self.data[dataset]

    def get(self, dataset, group, *args, **kwargs):
        if not self.has(dataset, group):
            import ROOT
            ROOT.PyConfig.IgnoreCommandLineOptions = True
            ROOT.gROOT.SetBatch()
            import pandas as pd
            import root_pandas

            # make sure all files are loaded
            self.load(dataset, group)

            def read(file_key):
                # load the file
                cache_path = self.get_cache_path(dataset, group, file_key)
                df = root_pandas.read_root(cache_path, *args, **kwargs)

                # convert uints to signed ints for safe comparison
                dtypes = {
                    col: dtype.name[1:]
                    for col, dtype in df.dtypes.items()
                    if dtype.name.startswith("uint")
                }
                if dtypes:
                    df = df.astype(dtypes)

                # apply a transformation?
                transformation = self.config.get_transformation(dataset, group)
                if transformation:
                    func = getattr(transformations, transformation)
                    df = func(df, dataset, group, file_key)

                return df

            # read datafiles for all file keys and combine them
            df = pd.concat([read(file_key) for file_key in self.config.get_files(dataset, group)])

            # save the dataframe
            self.data[dataset][group] = df

        return self.data[dataset][group]

    def load(self, dataset=None, group=None, flush=False, ci=False, *args, **kwargs):
        # flush first?
        if flush:
            self.flush(dataset, group)

        # prepare combinations of groups and datasets
        for dataset, group in self.config.get_dataset_group_combinations(dataset, group):
            # loop through all files
            for file_key, file_path in self.config.get_files(dataset, group).items():
                tpl = (dataset, group, file_key, file_path)
                cache_path = self.get_cache_path(*tpl[:3])

                # do nothing when already loaded
                if os.path.exists(cache_path):
                    continue

                # download or copy the data to the data dir
                if file_path.startswith("http://") or file_path.startswith("https://"):
                    print("download dataset {0} dataset ({2}) for group {1} from {3}".format(*tpl))
                    urllib.urlretrieve(file_path, cache_path)

                elif file_path.startswith(("/afs/", "/eos/")) and ci:
                    cmd = self.SCP_TMPL.format(
                        user=os.environ.get("KRB_USER", getpass.getuser()),
                        remote_path=file_path, local_path=cache_path)
                    print("scp dataset {0} ({2}) for group {1} from {3}".format(*tpl))
                    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
                    p.communicate()
                    if p.returncode != 0:
                        raise Exception("scp failed with code {}, command:\n{}".format(
                            p.returncode, cmd))

                else:
                    print("copy dataset {0} ({2}) for group {1} from {3}".format(*tpl))
                    shutil.copy2(file_path, cache_path)

                # change permissions
                os.chmod(cache_path, 0o0664)

    def flush(self, dataset=None, group=None):
        for dataset, group in self.config.get_dataset_group_combinations(dataset, group):
            for file_key in self.config.get_files(dataset, group):
                cache_path = self.get_cache_path(dataset, group, file_key)
                if os.path.exists(cache_path):
                    print("removed cached file for dataset {} ({}) and group {}".format(
                        dataset, file_key, group))
                    os.remove(cache_path)
