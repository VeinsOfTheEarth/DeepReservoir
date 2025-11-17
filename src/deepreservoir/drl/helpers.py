import pandas as pd

def split_train_test_by_water_year(df: pd.DataFrame, n_years_test: int = 10):
    """
    Split df into train/test by *water year* (Oct–Sep), taking the last
    `n_years_test` water years as test.

    Returns
    -------
    train_df, test_df
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex on df.")

    # USGS-style water year: Oct–Sep
    # e.g. 2015-10-01 → WY 2016, 2016-09-30 → WY 2016
    wy = pd.Series(idx.year + (idx.month >= 10).astype(int), index=idx, name="water_year")

    last_wy = int(wy.max())
    first_test_wy = last_wy - n_years_test + 1

    test_mask = wy >= first_test_wy

    train_df = df.loc[~test_mask]
    test_df  = df.loc[test_mask]

    print(
        f"[model_data] train {train_df.index.min().date()}–{train_df.index.max().date()}, "
        f"test {test_df.index.min().date()}–{test_df.index.max().date()} "
        f"({n_years_test} water years test; split between WY {first_test_wy-1} and {first_test_wy})."
    )

    return train_df, test_df
