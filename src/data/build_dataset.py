import numpy as np
import os
import re
import glob
import logging
import requests
import zipfile
import pandas as pd
import utils.constants as c
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_zip_links(url, station_ids, product_code):
    """Scrape the given URL for zip file links matching the product code and station IDs."""
    
    logging.info(f"Fetching zip file links from {url} for product '{product_code}'")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".zip") and product_code in href:
            m = re.search(rf"_{product_code}_([0-9]{{5}})_", href)
            if m:
                sid = m.group(1)
                if sid in station_ids:
                    full_link = url + href
                    links.append(full_link)
    logging.info(f"Found {len(links)} zip files for product '{product_code}'")
    return links


def download_zip_file(url, save_dir):
    """Download a zip file from a URL if it does not exist already."""

    os.makedirs(save_dir, exist_ok=True)
    local_filename = os.path.join(save_dir, url.split("/")[-1])
    if os.path.exists(local_filename):
        logging.info(f"File '{local_filename}' already exists. Skipping download.")
        return local_filename

    logging.info(f"Downloading '{url}' to '{local_filename}'")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            f.write(r.content)
        logging.info(f"Downloaded '{local_filename}'")
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return None


def process_zip_file(zip_file_path, product):
    """Extract and process a zip file. Reads the .txt file inside and renames columns."""

    df_list = []
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for filename in z.namelist():
            if filename.endswith(".txt"):
                with z.open(filename) as f:
                    if product == "wind":
                        df = pd.read_csv(f, sep=";", encoding="latin1")
                        df = df.rename(
                            columns={
                                "STATIONS_ID": "station_id",
                                "MESS_DATUM": "datetime",
                                "FF_10": "wind_speed",
                                "DD_10": "wind_direction",
                            }
                        )
                        df = df[["station_id", "datetime", "wind_speed", "wind_direction"]]
                    elif product == "TU":
                        df = pd.read_csv(f, sep=";", encoding="latin1")
                        df = df.rename(
                            columns={
                                "STATIONS_ID": "station_id",
                                "MESS_DATUM": "datetime",
                                "PP_10": "air_pressure",
                                "TT_10": "air_temperature",
                                "RF_10": "relative_humidity",
                                "TD_10": "dew_point",
                            }
                        )
                        df = df[
                            [
                                "station_id",
                                "datetime",
                                "air_pressure",
                                "air_temperature",
                                "relative_humidity",
                                "dew_point",
                            ]
                        ]
                    df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


def process_meta_zip_file(zip_file_path):
    """Extracts and processes metadata from a zip file."""

    with zipfile.ZipFile(zip_file_path, "r") as z:
        for filename in z.namelist():
            if filename.startswith("Metadaten_Geographie_") and filename.endswith(".txt"):
                with z.open(filename) as f:
                    df = pd.read_csv(f, sep=";", encoding="latin1")
                    df = df.rename(
                        columns={
                            "Stations_id": "id",
                            "Stationshoehe": "altitude",
                            "Geogr.Breite": "geographic_latitude",
                            "Geogr.Laenge": "geographic_longitude",
                            "Stationsname": "name",
                            "von_datum": "valid_from",
                            "bis_datum": "valid_until",
                        }
                    )
                    return df
    return pd.DataFrame()


def filter_stations(df, station_ids):
    """Filter the DataFrame to only include rows for the specified station IDs."""

    df["station_id"] = df["station_id"].astype(str).str.zfill(5)
    return df[df["station_id"].isin(station_ids)]


def load_and_process(product, download_folder, station_ids):
    """Load and process all zip files for the station_ids for a given product from a folder."""

    all_zip_files = glob.glob(os.path.join(download_folder, "*.zip"))
    zip_files_for_stations = [
        file
        for file in all_zip_files
        if any(f"_{station_id}_" in os.path.basename(file) for station_id in station_ids)
    ]

    df_list = []
    for file in zip_files_for_stations:
        logging.info(f"Processing file: {file}")
        df = process_zip_file(file, product)
        if not df.empty:
            df = filter_stations(df, station_ids)
            df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


def build_metadata_dataset(meta_download_dir, station_ids, output_path, max_end_date):
    """Combines the most recent metadata for each station and saves as a CSV."""

    all_zip_files = glob.glob(os.path.join(meta_download_dir, "*.zip"))
    zip_files_for_stations = [
        file
        for file in all_zip_files
        if any(f"_{station_id}" in os.path.basename(file) for station_id in station_ids)
    ]

    meta_df_list = []
    for file in zip_files_for_stations:
        logging.info(f"Processing metadata file: {file}")
        df = process_meta_zip_file(file)
        if not df.empty:
            meta_df_list.append(df)

    if meta_df_list:
        combined_df = pd.concat(meta_df_list, ignore_index=True)

        combined_df["valid_from"] = pd.to_datetime(
            combined_df["valid_from"], format="%Y%m%d", errors="coerce"
        )
        combined_df["valid_until"] = pd.to_datetime(
            combined_df["valid_until"], format="%Y%m%d", errors="coerce"
        )

        # empty values mean there is currently no restriction to what end the station is valid
        combined_df["valid_until"].fillna(max_end_date, inplace=True)

        combined_df["id"] = combined_df["id"].astype(str).str.zfill(5)
        combined_df = combined_df[combined_df["id"].isin(station_ids)]

        group_cols = [
            col for col in combined_df.columns if col not in ["valid_from", "valid_until"]
        ]

        # if metadata does not change we can merge the entries together
        merged_df = combined_df.groupby(group_cols, as_index=False).agg(
            valid_from=("valid_from", "min"), valid_until=("valid_until", "max")
        )

        most_recent_merged_df = merged_df.sort_values("valid_from").groupby("id").tail(1)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        most_recent_merged_df.to_csv(output_path, index=False)
        logging.info(f"Station metadata saved to '{output_path}'")


def get_starting_and_end_date(metadata_output_path):
    """Extracts the common time range for all stations based on metadata file."""

    metadata_df = pd.read_csv(metadata_output_path)

    metadata_df["valid_from"] = pd.to_datetime(metadata_df["valid_from"], errors="coerce")
    metadata_df["valid_until"] = pd.to_datetime(metadata_df["valid_until"], errors="coerce")

    latest_starting_date = metadata_df["valid_from"].max()
    earliest_end_date = metadata_df["valid_until"].min()

    return (latest_starting_date, earliest_end_date)


def build_complete_dataset(
    wind_download_dir,
    temperature_download_dir,
    dataset_output_path,
    metadata_output_path,
    station_ids,
):
    """Builds a complete dataset by merging wind and temperature data and applying time filtering."""

    logging.info("Loading wind data...")
    wind_df = load_and_process("wind", wind_download_dir, station_ids)
    logging.info("Loading temperature data...")
    temperature_df = load_and_process("TU", temperature_download_dir, station_ids)

    for df in [wind_df, temperature_df]:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M", errors="coerce")

    merged_df = pd.merge(wind_df, temperature_df, on=["station_id", "datetime"], how="outer")
    merged_df.sort_values(by=["station_id", "datetime"], inplace=True)

    (start_date, end_date) = get_starting_and_end_date(metadata_output_path)
    station_validity_mask = (merged_df["datetime"] >= start_date) & (
        merged_df["datetime"] <= end_date
    )

    merged_df = merged_df.loc[station_validity_mask]

    pivot_df = merged_df.pivot_table(
        index="datetime",
        columns="station_id",
        values=[
            "wind_speed",
            "wind_direction",
            "air_pressure",
            "air_temperature",
            "relative_humidity",
            "dew_point",
        ],
        aggfunc="first",
    )

    pivot_df.columns = [f"{measurement}_{station}" for measurement, station in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    output_dir = os.path.dirname(dataset_output_path)
    os.makedirs(output_dir, exist_ok=True)
    pivot_df.to_csv(dataset_output_path, index=False)
    logging.info(f"Complete dataset saved to '{dataset_output_path}'")


def insertNaNs(output_path):
    """Inserts NaNs for values that equal -999.0 (standard described by the DWD)."""

    df = pd.read_csv(output_path)
    df.replace(-999.0, np.nan, inplace=True)
    df.to_csv(output_path, index=False)


def main():

    meta_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/meta_data/"
    meta_download_dir = os.path.join("data", "downloads", "meta_data")
    wind_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/historical/"
    temperature_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/historical/"
    wind_download_dir = os.path.join("data", "downloads", "wind")
    temperature_download_dir = os.path.join("data", "downloads", "air_temperature")
    metadata_output_path = os.path.join("data", "raw", "station_metadata.csv")
    dataset_output_path = os.path.join("data", "raw", "complete_dataset.csv")

    max_end_date = pd.to_datetime("2023-12-31")

    os.makedirs(meta_download_dir, exist_ok=True)
    meta_zip_links = [
        f"{meta_url}Meta_Daten_zehn_min_ff_{station_id}.zip" for station_id in c.STATION_IDS
    ]
    for url in meta_zip_links:
        download_zip_file(url, meta_download_dir)

    logging.info("Building station metadata dataset...")
    build_metadata_dataset(meta_download_dir, c.STATION_IDS, metadata_output_path, max_end_date)

    os.makedirs(wind_download_dir, exist_ok=True)
    os.makedirs(temperature_download_dir, exist_ok=True)

    wind_zip_links = get_zip_links(wind_url, c.STATION_IDS, product_code="wind")
    temperature_zip_links = get_zip_links(temperature_url, c.STATION_IDS, product_code="TU")

    for url in wind_zip_links:
        download_zip_file(url, wind_download_dir)
    for url in temperature_zip_links:
        download_zip_file(url, temperature_download_dir)

    build_complete_dataset(
        wind_download_dir,
        temperature_download_dir,
        dataset_output_path,
        metadata_output_path,
        c.STATION_IDS,
    )

    insertNaNs(dataset_output_path)


if __name__ == "__main__":
    main()
