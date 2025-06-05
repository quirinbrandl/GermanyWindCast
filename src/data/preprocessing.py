from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import utils.constants as c
from utils.data_utils import get_time_steps, load_dataset


def remove_leading_trailing_nans(df):
    """Remove all leading and trailing rows of the data frame that contain NaN values."""

    non_missing_dates = df[df.notna().all(axis=1)].index

    first_non_missing_date = non_missing_dates.min()
    last_non_missing_date = non_missing_dates.max()

    return df[(df.index >= first_non_missing_date) & (df.index <= last_non_missing_date)]


def compute_valid_idxs(df, max_look_back_steps, max_forecasting_steps):
    """
    Generate inclusive ending indices of all look back windows that itself
    and their forecasting horizon do not cotain NaNs.
    """

    total_window_size = max_forecasting_steps + max_look_back_steps
    clean_window = (
        df.isna()
        .any(axis=1)
        .rolling(total_window_size, min_periods=total_window_size)
        .sum()
        .eq(0)
    )
    end_mask = clean_window.shift(-max_forecasting_steps, fill_value=False)

    return np.flatnonzero(end_mask)


def split_idxs(idxs, train_ratio, eval_ratio, max_look_back_steps, max_forecasting_steps):
    num_samples = len(idxs)
    full_window_length = max_look_back_steps + max_forecasting_steps
    num_possibly_overlapping_samples = full_window_length - 1

    # possibly overlapping samples are not used to prevent data leakage
    num_used_samples = num_samples - (2 * num_possibly_overlapping_samples)

    train_size = int(train_ratio * num_used_samples)
    eval_size = int(eval_ratio * num_used_samples)

    train_idxs = idxs[:train_size]

    eval_start = train_size + num_possibly_overlapping_samples
    eval_end = eval_start + eval_size
    eval_idxs = idxs[eval_start:eval_end]

    test_idxs = idxs[eval_end + num_possibly_overlapping_samples :]

    return train_idxs, eval_idxs, test_idxs


def save_idxs(train_idxs, eval_idxs, test_idxs, output_dir):
    pd.Series(train_idxs).to_csv(
        output_dir / "train_indices.csv", index=False, header=["valid_indices"]
    )
    pd.Series(eval_idxs).to_csv(
        output_dir / "eval_indices.csv", index=False, header=["valid_indices"]
    )
    pd.Series(test_idxs).to_csv(
        output_dir / "test_indices.csv", index=False, header=["valid_indices"]
    )


def add_temporal_features(df):
    """Add temporal features with 1 year, 1 day and 34 day periodicity."""

    df_temporal = df.copy()

    tsec = df.index.view("int64") / 1e9
    sec_1d = 60**2 * 24

    df_temporal["sin_1y"] = np.sin((2 * np.pi * tsec) / (365.25 * sec_1d)) ## account for leap years
    df_temporal["cos_1y"] = np.cos((2 * np.pi * tsec) / (365.25 * sec_1d))

    df_temporal["sin_34d"] = np.sin((2 * np.pi * tsec) / (34.7 * sec_1d))
    df_temporal["cos_34d"] = np.cos((2 * np.pi * tsec) / (34.7 * sec_1d))

    df_temporal["sin_1d"] = np.sin((2 * np.pi * tsec) / 86400)
    df_temporal["cos_1d"] = np.cos((2 * np.pi * tsec) / 86400)

    return df_temporal


def transform_wind_direction(df):
    """Transform wind direction into sine and cosine features."""

    df_wind_dir = df.copy()

    for station_id in c.STATION_IDS:
        direction_col = f"wind_direction_{station_id}"
        direction_rads = np.deg2rad(df[direction_col])

        df_wind_dir[f"wind_direction_sin_{station_id}"] = np.sin(direction_rads)
        df_wind_dir[f"wind_direction_cos_{station_id}"] = np.cos(direction_rads)

        df_wind_dir.drop(columns=[direction_col], inplace=True)

    return df_wind_dir


def get_feature_from_col(col):
    feature = col if not col.endswith(tuple(c.STATION_IDS)) else col.rsplit("_", 1)[0]
    return feature


def scale_features_globally(df, eval_start_idx):
    df_scaled = df.copy()

    training_data = df.iloc[:eval_start_idx]
    scaler_dict = {}

    for col in df.columns:
        feature = get_feature_from_col(col)
        scaler_dict.setdefault(feature, StandardScaler())

    for feature, scaler in scaler_dict.items():
        cols = [col for col in df.columns if col.startswith(f"{feature}")]

        stacked = training_data[cols].values.reshape(-1, 1)
        scaler.fit(stacked)

    for col in df.columns:
        feature = get_feature_from_col(col)
        scaler = scaler_dict[feature]
        df_scaled.loc[:, col] = scaler.transform(df[col].values.reshape(-1, 1)).ravel()

    return df_scaled, scaler_dict


def scale_features_locally(df, eval_start_idx):
    df_scaled = df.copy()

    training_data = df.iloc[:eval_start_idx]
    scaler_dict = {}

    for col in df.columns:
        scaler_dict[col] = StandardScaler().fit(training_data[col].values.reshape(-1, 1))

    for col, scaler in scaler_dict.items():
        df_scaled.loc[:, col] = scaler.transform(df[col].values.reshape(-1, 1)).ravel()

    return df_scaled, scaler_dict


def precompute_rolling_mean_targets(df, target_resolution):
    df_rolling_mean = df.copy()

    window_size = get_time_steps(target_resolution)
    rolling_mean = df[f"wind_speed_{c.REFERENCE_STATION_ID}"].rolling(window_size).mean()
    rolling_mean = rolling_mean.shift(-(window_size - 1))

    df_rolling_mean["target"] = rolling_mean
    return df_rolling_mean


def main():
    """Main preprocessing pipeline."""

    print("Loading data...")
    dataset = load_dataset()

    print("Removing leading and trainling NaNs...")
    dataset = remove_leading_trailing_nans(dataset)

    print("Adding temporal features...")
    dataset = add_temporal_features(dataset)

    print("Transforming wind direction...")
    dataset = transform_wind_direction(dataset)

    print("Calculating valid sliding window indices...")
    max_look_back_steps = get_time_steps("12h")
    max_forecasting_steps = get_time_steps("8h")
    valid_idxs = compute_valid_idxs(dataset, max_look_back_steps, max_forecasting_steps)

    print("Splitting the idxs in train/test/eval sets...")
    train_idxs, eval_idxs, test_idxs = split_idxs(
        valid_idxs,
        max_look_back_steps=max_look_back_steps,
        max_forecasting_steps=max_forecasting_steps,
        train_ratio=0.7,
        eval_ratio=0.15,
    )

    print("Normalizing features...")
    dataset_gl_scaled, global_scalers = scale_features_globally(dataset, eval_idxs[0])
    dataset_lc_scaled, local_scalers = scale_features_locally(dataset, eval_idxs[0])

    idx_lengths = [len(idxs) for idxs in [train_idxs, eval_idxs, test_idxs]]
    total_size = sum(idx_lengths)
    train_size, eval_size, test_size = idx_lengths
    train_percent, eval_percent, test_percent = [
        (idx_length / total_size) * 100 for idx_length in idx_lengths
    ]

    summary = (
        f"In total, there are {total_size} used starting indices for a look-back window.\n"
        f"- {train_size} ({train_percent:.2f}%) in the training set\n"
        f"- {eval_size} ({eval_percent:.2f}%) in the evaluation set\n"
        f"- {test_size} ({test_percent:.2f}%) in the test set."
    )

    print(summary)

    print("Precomputing rolling mean for targets...")
    dataset_gl_scaled_with_targets = precompute_rolling_mean_targets(dataset_gl_scaled, "1h")
    dataset_lc_scaled_with_targets = precompute_rolling_mean_targets(dataset_lc_scaled, "1h")

    print("Save results...")
    output_dir = Path(f"data/processed/")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_gl_scaled_with_targets.to_csv(output_dir / "dataset_gl_scaled.csv")
    dataset_lc_scaled_with_targets.to_csv(output_dir / "dataset_lc_scaled.csv")

    save_idxs(train_idxs, eval_idxs, test_idxs, output_dir)

    joblib.dump(global_scalers, output_dir / "global_scalers.pkl")
    joblib.dump(local_scalers, output_dir / "local_scalers.pkl")

    print(f"Preprocessing completed. Everything saved to {output_dir}.")


if __name__ == "__main__":
    main()
