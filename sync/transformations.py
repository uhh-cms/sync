# coding: utf-8

"""
Group-specific transformations that act on pandas data frames to work with identical variables.
"""


from sync.tools import MISSING


def example(df, group, dataset):
    # if you want to rename a column inplace, just do
    # df.rename(columns={"old": "new"}, inplace=True)

    return df


def transform_cern_2018(df, group, dataset, file_key):
    # some selections
    df = df[
        # at least two b-tagged jets
        df.eval("((n_btags >= 2))")
    ]

    # add a "pairType" column (similar to "channel" anyway)
    # mutau: 2 -> 0, eta: 1 -> 1 (nothing to do), tautau: 3 -> 2
    pair_type = df["channel"].values
    pair_type[pair_type == 2] = 0
    pair_type[pair_type == 3] = 2
    df.insert(len(df.columns), "pairType", pair_type)

    # add a "isOS" column using the "region"
    # 1|2 -> OS, 3|4 -> SS
    is_os = df["region"].values
    is_os[(is_os == 1) | (is_os == 2)] = 1
    is_os[(is_os == 3) | (is_os == 4)] = 0
    df.insert(len(df.columns), "isOS", is_os)

    return df


def transform_mibllr_2018(df, group, dataset, file_key):
    import numpy as np

    # some selections
    df = df[
        # mutau or eta or tautau
        df.eval("((pairType == 0) | (pairType == 1) | (pairType == 2))")
        # no additional leptons
        & df.eval("((nleps == 0))")
        # at least two b-tagged jets
        & df.eval("((nbjetscand >= 2))")
    ]

    # add flat per-jet columns
    n = 2
    spec = {
        "jets_pt": "jet{}_pt",
        "jets_eta": "jet{}_eta",
        "jets_phi": "jet{}_phi",
    }
    jet_vars = {var_name: [np.zeros(len(df)) for _ in range(n)] for var_name in spec}
    for i in range(len(df)):
        values = {var_name: df[var_name].values for var_name in spec}
        for var_name, var_list in jet_vars.items():
            for j, arr in enumerate(var_list):
                if len(values[var_name][i]) - 1 > j:
                    arr[i] = values[var_name][i][j]
                else:
                    arr[i] = MISSING
    # insert into dataframe
    for var_name, tmplate in spec.items():
        for i in range(n):
            df.insert(len(df.columns), tmplate.format(i + 1), jet_vars[var_name][i])

    # finally rename some columns, "old" -> "new"
    df = df.rename(columns={
        "RunNumber": "run",
        "EventNumber": "event",
        "npu": "pu",
        "njets20": "n_jets",
        "nbjets20": "n_btags",
        "dau1_pt": "tau1_pt",
        "dau1_eta": "tau1_eta",
        "dau1_phi": "tau1_phi",
        "dau1_decayMode": "tau1_decay_mode",
    })

    return df


def transform_pikosi_2018(df, group, dataset, file_key):
    import numpy as np

    # some selections
    df = df[
        # at least two b-tagged jets
        df.eval("((nbjets >= 2))")
    ]

    # add a "pairType" column, base on the file_key
    # mutau -> 0, etau -> 1, tautau -> 2
    pair_type = np.ones(len(df)) * {"mutau": 0, "etau": 1, "tautau": 2}[file_key]
    df.insert(len(df.columns), "pairType", pair_type)

    # add a "isOS" column using the leg charges
    is_os = df.eval("(q_1 != q_2)").values.astype(int)
    df.insert(len(df.columns), "isOS", is_os)

    # finally rename some columns, "old" -> "new"
    df = df.rename(columns={
        "evt": "event",
        "npu": "pu",
        "njets": "n_jets",
        "nbjets": "n_btags",
        "pt_1": "tau1_pt",
        "eta_1": "tau1_eta",
        "phi_1": "tau1_phi",
        "decayModeFindingOldDMs_1": "tau1_decay_mode",
        "jpt_1": "jet1_pt",
        "jeta_1": "jet1_eta",
        "jphi_1": "jet1_phi",
        "jpt_2": "jet2_pt",
        "jeta_2": "jet2_eta",
        "jphi_2": "jet2_phi",
    })

    return df
