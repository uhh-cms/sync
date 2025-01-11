# coding: utf-8

"""
Group-specific transformations that act on pandas dataframes to convert them into a common format.
Function names can be referenced as strings in config files.
"""

from __future__ import annotations

__all__ = ["example"]

import pandas as pd


def example(dataset: str, group: str, file_key: int | str, df: pd.DataFrame) -> pd.DataFrame:
    # if you want to rename a column inplace, just do
    # df.rename(columns={"old": "new"}, inplace=True)

    return df


def cclub_to_cf(dataset: str, group: str, file_key: int | str, df: pd.DataFrame) -> pd.DataFrame:
    # re-map channel_ids
    cclub_channels = {"mutau": 0, "etau": 1, "tautau": 2, "mumu": 3, "ee": 4, "emu": 5}
    cf_channels = {"etau": 1, "mutau": 2, "tautau": 3, "ee": 4, "mumu": 5, "emu": 6}
    club_to_cf_map = {cclub_channels[k]: cf_channels[k] for k in cclub_channels.keys()}
    df["channel_id"] = df["channel_id"].map(club_to_cf_map)

    return df
