import pandas as pd
import numpy as np
import requests
from pathlib import Path
import utils.constants as c
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

OPEN_METEO_COL_NAME_TO_DATASET_COL_NAME_MAPPING = {
    "temperature_2m": "air_temperature",
    "relative_humidity_2m": "relative_humidity",
    "dew_point_2m": "dew_point",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "surface_pressure": "air_pressure",
}


def load_data():
    """Load the raw dataset and station metadata."""

    data_path = Path("data/raw/complete_dataset.csv")
    metadata_path = Path("data/raw/station_metadata.csv")

    df = pd.read_csv(data_path, parse_dates=True, index_col="datetime")
    metadata = pd.read_csv(metadata_path, dtype={"id": str})

    return df, metadata


def build_different_time_res(df_10min):
    """Build datasets with 20min, 30min and 1h time resolution."""
    time_resolutions = ["20min", "30min", "1h"]

    datasets = {"10min": df_10min}
    for res in time_resolutions:
        datasets[res] = df_10min.resample(res, closed="right", label="right").mean()

    return datasets


def get_openmeteo_data(df_metadata, start_date, end_date):
    """
    Fetch hourly historical weather data from OpenMeteo API for all stations specified in the
    metadata dataframe.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": df_metadata["geographic_latitude"],
        "longitude": df_metadata["geographic_longitude"],
        "elevation": df_metadata["altitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": OPEN_METEO_COL_NAME_TO_DATASET_COL_NAME_MAPPING.keys(),
        "timezone": "Europe/Berlin",
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        time_values = data[0]["hourly"]["time"]
        data_dict = {}

        for i, station in enumerate(data):
            for (
                openmeteo_col_name,
                dataset_col_name,
            ) in OPEN_METEO_COL_NAME_TO_DATASET_COL_NAME_MAPPING.items():
                column_name = f"{dataset_col_name}_{c.STATION_IDS[i]}"
                measurement_data = station["hourly"][openmeteo_col_name]
                data_dict[column_name] = measurement_data

        openmeteo_df = pd.DataFrame(data_dict, index=pd.to_datetime(time_values))
        openmeteo_df.index.name = "datetime"

        return openmeteo_df
    else:
        print(response)
        raise Exception(f"Error when accessing OpenMeteo's API: {response.json()}")


def remove_leading_trailing_nans(df):
    """Remove all leading and trailing rows of the data frame that contain NaN values."""

    non_missing_dates = df[df.notna().all(axis=1)].index

    first_non_missing_date = non_missing_dates.min()
    last_non_missing_date = non_missing_dates.max()

    return df[(df.index >= first_non_missing_date) & (df.index <= last_non_missing_date)]


def enforce_hourly_start_end_date(df):
    """
    If not already the case this method cuts the dataset such that the clock of the start date is
    hh:10:00 and the clock of the end_date is hh:00:00. This ensures that the dataset starts and
    ends at full hour since the measurements in the raw dataset are always the average measured in
    the previous 10 minutes
    """

    current_start_date = df.index.min()
    current_end_date = df.index.max()

    full_hour_start_date = current_start_date.ceil("h") + pd.Timedelta(minutes=10)
    full_hour_end_date = current_end_date.floor("h")

    start_end_at_full_hour_mask = (df.index >= full_hour_start_date) & (
        df.index <= full_hour_end_date
    )

    return df[start_end_at_full_hour_mask]


def fill_missing_values(df, metadata):
    """
    Fill missing values in 10-minute resolution data using hourly data from OpenMeteo and
    interpolating the rest linearly.
    """
    df_filled = df.copy()

    missing_dates = df[df.isna().any(axis=1)].index
    if missing_dates.empty:
        return df_filled

    start_date = missing_dates.min()
    end_date = missing_dates.max()

    print("Fetching missing data from OpenMeteo API...")
    openmeteo_df = get_openmeteo_data(
        df_metadata=metadata,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    """ openmeteo_df = pd.read_csv(Path("data/downloads/openmeteo.csv"), parse_dates=True, index_col="datetime") """
    df_filled = df_filled.fillna(openmeteo_df)
    df_filled = df_filled.interpolate("linear")

    remaining_missing = df_filled.isna().sum().sum()
    print(f"Missing values are filled. There are {remaining_missing} missing values left.")

    return df_filled


def add_temporal_features(df):
    """Add hour of day and day of year features."""

    df_temporal = df.copy()

    hours = df_temporal.index.hour
    df_temporal["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df_temporal["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    days = df_temporal.index.dayofyear
    df_temporal["day_sin"] = np.sin(
        2 * np.pi * days / 365.25  # 365.25 days to account for leap years
    )
    df_temporal["day_cos"] = np.cos(2 * np.pi * days / 365.25)

    return df_temporal


def transform_wind_direction(df):
    """Transform wind direction into sine and cosine features."""

    df_wind_vec = df.copy()

    for station_id in c.STATION_IDS:
        direction_col = f"wind_direction_{station_id}"
        direction_rads = np.deg2rad(df[direction_col])

        df_wind_vec[f"wind_direction_sin_{station_id}"] = np.sin(direction_rads)
        df_wind_vec[f"wind_direction_cos_{station_id}"] = np.cos(direction_rads)

        df_wind_vec.drop(columns=[direction_col], inplace=True)

    return df_wind_vec


def scale_features(df, scaler_dict=None):
    """Scale the features in the given df."""

    if scaler_dict == None:
        scaler_dict = {
            "wind_speed": StandardScaler(),
            "air_temperature": StandardScaler(),
            "air_pressure": StandardScaler(),
            "dew_point": StandardScaler(),
            "relative_humidity": MinMaxScaler(),
        }

    for col in df.columns:
        col_base_name = col[:-6]

        if col_base_name in scaler_dict.keys():
            scaler = scaler_dict[col_base_name]
            df.loc[:, col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df, scaler_dict


def get_split_times(df, train_size=0.7, val_size=0.15):
    """Compute cut-off timestamps (floored to the hour) for train and validation splits."""

    n = len(df)

    train_idx = int(n * train_size)
    val_idx   = int(n * (train_size + val_size))
    train_time = df.index[train_idx].floor("h")
    val_time   = df.index[val_idx].floor("h")

    return train_time, val_time


def split_dataset(df, train_time, val_time):
    """Slice df into train/val/test using given times."""

    train = df[df.index <= train_time]
    val   = df[(df.index > train_time) & (df.index <= val_time)]
    test  = df[df.index > val_time]

    return train, val, test


def get_number_of_nans(df):
    return df.isna().sum().sum()


def main():
    """Main preprocessing pipeline."""

    print("Loading data...")
    raw_df, metadata_df = load_data()

    print("Removing and fillin missing values...")
    no_nan_df = remove_leading_trailing_nans(raw_df)
    no_nan_df = fill_missing_values(no_nan_df, metadata_df)

    print("Enforcing hourly start and end dates...")
    cut_df = enforce_hourly_start_end_date(no_nan_df)

    print("Create different time resolution datasets...")
    datasets = build_different_time_res(cut_df)

    print("Computing single train/val cut-off times on 10 min resolution…")
    train_time, val_time = get_split_times(datasets["10min"])

    print(
        "Add temporal features, scale features, perform train/val/test splitting for all datasets..."
    )

    for resolution in datasets:
        dataset = datasets[resolution]

        dataset = add_temporal_features(dataset)
        dataset = transform_wind_direction(dataset)

        train, val, test = split_dataset(dataset, train_time, val_time)

        train, scalers = scale_features(train)
        val, _ = scale_features(val, scaler_dict=scalers)
        test, _ = scale_features(test, scaler_dict=scalers)

        output_dir = Path(f"data/processed/{resolution}res")
        output_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(output_dir / "train.csv")
        val.to_csv(output_dir / "eval.csv")
        test.to_csv(output_dir / "test.csv")

        joblib.dump(scalers, Path(output_dir / "scalers.pkl"))

    print("Preprocessing completed. Datasets saved to data/processed/")


if __name__ == "__main__":
    main()
