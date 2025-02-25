import os
import re
import glob
import logging
import requests
import zipfile
import pandas as pd
from bs4 import BeautifulSoup

"""
This script is responsible for downloading the raw data from DWD and building a single complete
dataset that later can be used for training, evaluating and testing differen models.
"""

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_zip_links(url, station_ids, product_code):
    """
    Scrape the given URL for zip file links matching the product code and station IDs.
    
    Parameters:
        url (str): The URL of the DWD directory.
        station_ids (list of str): The list of station IDs to include.
        product_code (str): The product code ('wind' or 'TU').
    
    Returns:
        list: A list of full URLs to zip files that match.
    """
    logging.info(f"Fetching zip file links from {url} for product '{product_code}'")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.zip') and product_code in href:
            m = re.search(rf'_{product_code}_([0-9]{{5}})_', href)
            if m:
                sid = m.group(1)
                if sid in station_ids:
                    full_link = url + href
                    links.append(full_link)
    logging.info(f"Found {len(links)} zip files for product '{product_code}'")
    return links

def download_zip_file(url, save_dir):
    """
    Download a zip file from a URL if it does not exist already.
    
    Parameters:
        url (str): URL to the zip file.
        save_dir (str): Directory to save the downloaded file.
    
    Returns:
        str or None: Local file path of the downloaded zip file, or None if download failed.
    """
    os.makedirs(save_dir, exist_ok=True)
    local_filename = os.path.join(save_dir, url.split('/')[-1])
    if os.path.exists(local_filename):
        logging.info(f"File '{local_filename}' already exists. Skipping download.")
        return local_filename
    
    logging.info(f"Downloading '{url}' to '{local_filename}'")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(r.content)
        logging.info(f"Downloaded '{local_filename}'")
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return None

def process_zip_file(zip_file_path, product):
    """
    Extract and process a zip file. Reads the .txt file inside and renames columns.
    
    Parameters:
        zip_file_path (str): Local path to the zip file.
        product (str): 'wind' for wind data or 'TU' for temperature data.
    
    Returns:
        pd.DataFrame: A DataFrame containing the relevant data.
    """
    df_list = []
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        for filename in z.namelist():
            if filename.endswith('.txt'):
                with z.open(filename) as f:
                    if product == 'wind':
                        df = pd.read_csv(f, sep=';', encoding='latin1')
                        df = df.rename(columns={
                            'STATIONS_ID': 'station_id',
                            'MESS_DATUM': 'datetime',
                            'FF_10': 'wind_speed',
                            'DD_10': 'wind_direction'
                        })
                        df = df[['station_id', 'datetime', 'wind_speed', 'wind_direction']]
                    elif product == 'TU':
                        df = pd.read_csv(f, sep=';', encoding='latin1')
                        df = df.rename(columns={
                            'STATIONS_ID': 'station_id',
                            'MESS_DATUM': 'datetime',
                            'PP_10': 'air_pressure',
                            'TT_10': 'air_temperature',
                            'RF_10': 'relative_humidity',
                            'TD_10': 'dew_point'
                        })
                        df = df[['station_id', 'datetime', 'air_pressure', 'air_temperature', 'relative_humidity', 'dew_point']]
                    df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def filter_stations(df, station_ids):
    """
    Filter the DataFrame to only include rows for the specified station IDs.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        station_ids (list of str): List of station IDs.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df['station_id'] = df['station_id'].astype(str).str.zfill(5)
    return df[df['station_id'].isin(station_ids)]

def load_and_process(product, download_folder, station_ids):
    """
    Load and process all zip files for a given product from a folder.
    
    Parameters:
        product (str): 'wind' or 'TU'.
        download_folder (str): Folder containing downloaded zip files.
        station_ids (list of str): List of station IDs to filter.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame for the product.
    """
    all_zip_files = glob.glob(os.path.join(download_folder, '*.zip'))
    df_list = []
    for file in all_zip_files:
        logging.info(f"Processing file: {file}")
        df = process_zip_file(file, product)
        if not df.empty:
            df = filter_stations(df, station_ids)
            df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    station_ids = ["05856", "03366", "03379", "04911", "04104", 
                   "06211", "03668", "05404", "05800", "13932"]

    wind_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/historical/"
    temperature_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/historical/"

    wind_download_dir = os.path.join('data', 'downloads', 'wind')
    temperature_download_dir = os.path.join('data', 'downloads', 'air_temperature')
    os.makedirs(wind_download_dir, exist_ok=True)
    os.makedirs(temperature_download_dir, exist_ok=True)

    wind_zip_links = get_zip_links(wind_url, station_ids, product_code='wind')
    temperature_zip_links = get_zip_links(temperature_url, station_ids, product_code='TU')

    for url in wind_zip_links:
        download_zip_file(url, wind_download_dir)
    for url in temperature_zip_links:
        download_zip_file(url, temperature_download_dir)

    logging.info("Loading wind data...")
    wind_df = load_and_process('wind', wind_download_dir, station_ids)
    logging.info("Loading temperature data...")
    temperature_df = load_and_process('TU', temperature_download_dir, station_ids)

    for df in [wind_df, temperature_df]:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M', errors='coerce')

    wind_df = wind_df[~((wind_df['station_id'] == "05800") & (wind_df['datetime'].dt.year < 2006))]
    temperature_df = temperature_df[~((temperature_df['station_id'] == "05800") & (temperature_df['datetime'].dt.year < 2006))]

    merged_df = pd.merge(wind_df, temperature_df, on=['station_id', 'datetime'], how='outer')
    merged_df.sort_values(by=['station_id', 'datetime'], inplace=True)

    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2022-12-31")
    mask = (merged_df['datetime'] >= start_date) & (merged_df['datetime'] <= end_date)
    merged_df = merged_df.loc[mask]
    
    pivot_df = merged_df.pivot_table(
        index='datetime',
        columns='station_id',
        values=['wind_speed', 'wind_direction', 'air_pressure', 'air_temperature', 'relative_humidity', 'dew_point'],
        aggfunc='first'
    )
    
    pivot_df.columns = [f"{measurement}_{station}" for measurement, station in pivot_df.columns]
    pivot_df.reset_index(inplace=True)
    
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'complete_raw_dataset_wide.csv')
    pivot_df.to_csv(output_path, index=False)
    logging.info(f"Complete wide-format dataset saved to '{output_path}'")

if __name__ == '__main__':
    main()