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
