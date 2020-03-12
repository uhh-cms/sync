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

from sync import translations


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

    def __getattr__(self, attr):
        if isinstance(attr, tuple) and len(attr) == 2:
            return self.data[attr[0]][attr[1]]
        else:
            return self.data[attr]

    def __contains__(self, pair):
        return self.has(*pair)

    def __call__(self, dataset, group):
        return self.get(dataset, group)

    def get_cache_path(self, dataset, group, ext="root"):
        return os.path.join(self.cli.cache_dir, "{}_{}.{}".format(dataset, group, ext))

    def has(self, dataset, group):
        return dataset in self.data and group in self.data[dataset]

    def get(self, dataset, group, *args, **kwargs):
        if not self.has(dataset, group):
            import ROOT
            ROOT.PyConfig.IgnoreCommandLineOptions = True
            ROOT.gROOT.SetBatch()
            import root_pandas

            # make sure the file is loaded
            self.load(dataset, group)

            # load the file
            cache_path = self.get_cache_path(dataset, group)
            df = root_pandas.read_root(cache_path, *args, **kwargs)

            # apply a translation?
            translation_name = self.config.get_translation(dataset, group)
            if translation_name:
                func = getattr(translations, translation_name)
                df = func(df, dataset, group)

            # save the dataframe
            self.data[dataset][group] = df

        return self.data[dataset][group]

    def load(self, dataset=None, group=None, flush=False, ci=False, *args, **kwargs):
        # flush first?
        if flush:
            self.flush(dataset, group)

        # prepare combinations of groups and datasets
        for pair in self.config.get_dataset_group_combinations(dataset, group):
            cache_path = self.get_cache_path(*pair)

            # do nothing when already loaded
            if os.path.exists(cache_path):
                continue

            # download or copy the data to the data dir
            remote_path = self.config.get_file(*pair)
            if remote_path.startswith("http://") or remote_path.startswith("https://"):
                print("download dataset {1} dataset for group {2} from {0}".format(
                    remote_path, *pair))
                urllib.urlretrieve(remote_path, cache_path)
            elif remote_path.startswith(("/afs/", "/eos/")) and ci:
                cmd = self.SCP_TMPL.format(
                    user=os.environ.get("KRB_USER", getpass.getuser()),
                    remote_path=remote_path, local_path=cache_path)
                print("scp dataset '{1}' for group '{2}' from {0}".format(remote_path, *pair))
                p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
                p.communicate()
                if p.returncode != 0:
                    raise Exception("scp failed with code {}, command:\n{}".format(
                        p.returncode, cmd))
            else:
                print("copy dataset {1} for group {2} from {0}".format(remote_path, *pair))
                shutil.copy2(remote_path, cache_path)

    def flush(self, dataset=None, group=None):
        for pair in self.config.get_dataset_group_combinations(dataset, group):
            cache_path = self.get_cache_path(*pair)
            if os.path.exists(cache_path):
                print("removed cached file for dataset '{}' and group '{}'".format(*pair))
                os.remove(cache_path)
