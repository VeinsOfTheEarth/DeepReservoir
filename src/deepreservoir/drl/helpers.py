from importlib import reload
import pandas as pd
import gymnasium as gym  # used for type hints once env is wired in
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


from deepreservoir.data import loader
from deepreservoir.drl import helpers


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


def load_datasets(n_years_test: int):
    """
    Load raw + normalized model data and split into train/test
    (using water-year-aware split).
    """
    reload(loader)  # convenient when iterating during dev
    nav_data = loader.NavajoData()
    alldata = nav_data.load_all(include_cont_streamflow=False, model_data=True)

    data = alldata["model_data"]              # raw
    datanorm = alldata["model_data_norm"]     # normalized
    norm_stats = alldata["model_norm_stats"]  # per-column mean/std

    # Split raw data into train/test by water year
    data_train, data_test = helpers.split_train_test_by_water_year(
        data, n_years_test=n_years_test
    )

    # Use the same index split for the normalized data
    datanorm_train = datanorm.loc[data_train.index]
    datanorm_test = datanorm.loc[data_test.index]

    return {
        "train_raw": data_train,
        "test_raw": data_test,
        "train_norm": datanorm_train,
        "test_norm": datanorm_test,
        "norm_stats": norm_stats,
    }


def build_reward(name: str, norm_stats):
    """
    Placeholder for a modular reward factory.

    Eventually this will look up `name` in a registry and return
    a callable or object that the environment can use to compute
    rewards (e.g., combining storage / fisheries / hydropower / NIIP).

    For now, just return a dummy function.
    """
    def dummy_reward(*args, **kwargs) -> float:
        return 0.0

    return dummy_reward



