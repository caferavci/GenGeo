"""
NYC Open Data Pipeline with Robust API Handling
Handles large datasets from:
1. DOT Traffic Speeds: https://data.cityofnewyork.us/resource/i4gi-tjb9.json
2. 311 Service Requests: https://data.cityofnewyork.us/resource/erm2-nwe9.json
3. NYC Permitted Event Information(Current): https://data.cityofnewyork.us/resource/tvpp-9vvx.json
4. NYC Permitted Event Information(Hisotical): https://data.cityofnewyork.us/resource/bkfu-528j.json
5. Automated Traffic Volume Counts: https://data.cityofnewyork.us/resource/7ym2-wayt.json
"""

import pandas as pd
import numpy as np
import requests
import json
from tqdm import tqdm
import time
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_CONFIG = {
    'traffic': {
        'url': 'https://data.cityofnewyork.us/resource/i4gi-tjb9.json',
        'date_field': 'data_as_of',
        'default_limit': 50000,
        'nested_fields': []
    },
    # '311': { # not use currently 
    #     'url': 'https://data.cityofnewyork.us/resource/erm2-nwe9.json',
    #     'date_field': 'created_date',
    #     'default_limit': 50000,
    #     'nested_fields': ['location']
    # },
    'events_current': {
        'url': 'https://data.cityofnewyork.us/resource/tvpp-9vvx.json',
        'date_field': 'start_date_time',
        'default_limit': 50000,
        'nested_fields': []
    },
     'events_historical': {
        'url': 'https://data.cityofnewyork.us/resource/bkfu-528j.json',
        'date_field': 'start_date_time',
        'default_limit': 50000,
        'nested_fields': []
    },
    'volume': {
        'url': 'https://data.cityofnewyork.us/resource/7ym2-wayt.json',
        'date_field': None, 
        'default_limit': 50000,
        'nested_fields': []
    }
}

def fetch_paginated_data(api_type: str, date_filter: tuple = None, max_records: int = None) -> pd.DataFrame:
    """
    Fetches data from SODA API endpoints with pagination, error handling, and retries.
    Skips date filtering and ordering for the 'volume' dataset via API.
    """
    if api_type not in API_CONFIG:
        logging.error(f"API configuration for '{api_type}' not found.")
        return pd.DataFrame()

    config = API_CONFIG[api_type]
    base_url = config['url']
    limit = config['default_limit']
    date_field = config.get('date_field') 
    nested_fields = config.get('nested_fields', [])

    params = {
        '$limit': limit,
        '$select': '*'
    }

    if date_field and api_type != 'volume':
         params['$order'] = f"`{date_field}` DESC"
    elif date_field and api_type == 'volume':
         logging.info(f"Skipping ordering ($order) for '{api_type}' dataset via API.")
    elif not date_field:
         logging.debug(f"No date_field configured for ordering '{api_type}'.")

    if date_filter and date_field and api_type != 'volume':
        start_date, end_date = date_filter
        try:
             start_dt_str = f"{datetime.strptime(start_date, '%Y-%m-%d').date()}T00:00:00.000"
             end_dt_str = f"{datetime.strptime(end_date, '%Y-%m-%d').date()}T23:59:59.999"
             params['$where'] = f"`{date_field}` >= '{start_dt_str}' AND `{date_field}` <= '{end_dt_str}'"
        except ValueError:
             logging.error(f"Invalid date format in date_filter for {api_type}. Expected YYYY-MM-DD.")
             return pd.DataFrame()
    elif date_filter and api_type == 'volume':
        logging.info(f"Skipping date filter ($where) for '{api_type}' dataset via API.")
    elif date_filter and not date_field:
        logging.warning(f"Date filter provided but no 'date_field' for '{api_type}'. Fetching without API date filter.")

    all_data = []
    offset = 0
    total_fetched = 0
    max_retries = 3
    current_retries = max_retries

    pbar_desc = f"Fetching {api_type} data"
    pbar_total = max_records if max_records else None
    if max_records: pbar_desc += f" (max {max_records})"

    with tqdm(total=pbar_total, desc=pbar_desc, unit=' records') as pbar:
        while True:
            try:
                params['$offset'] = offset
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()

                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                     logging.error(f"Failed to decode JSON response for {api_type}: {e}")
                     break

                if not data: break

                chunk_df = pd.DataFrame(data)
                chunk_processed = process_nested_data(chunk_df, nested_fields)

                if chunk_processed.empty: break

                all_data.append(chunk_processed)
                fetched = len(chunk_processed)
                total_fetched += fetched
                offset += fetched
                pbar.update(fetched)

                if max_records and total_fetched >= max_records:
                    logging.info(f"Reached max records ({max_records}) for {api_type}.")
                    break
                if fetched < limit: break

                time.sleep(0.5)
                current_retries = max_retries

            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed for {api_type} (offset {offset}): {e}")
                current_retries -= 1
                if current_retries > 0:
                    wait_time = 5 * (max_retries - current_retries + 1)
                    logging.info(f"Retrying in {wait_time} seconds... ({current_retries} retries left)")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Max retries exceeded for {api_type}. Stopping fetch.")
                    break
            except Exception as e:
                 logging.error(f"Unexpected error during fetch for {api_type}: {e}", exc_info=True)
                 break

    if not all_data:
        logging.warning(f"No data successfully fetched for {api_type}. Returning empty DataFrame.")
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    if max_records and len(full_df) > max_records:
        full_df = full_df.iloc[:max_records]
    return optimize_data_types(full_df, config.get('date_field'))


def process_nested_data(df: pd.DataFrame, nested_fields: list) -> pd.DataFrame:
    if df.empty or not nested_fields: return df
    df_copy = df.copy()
    for field in nested_fields:
        if field in df_copy.columns:
            if df_copy[field].apply(lambda x: isinstance(x, dict)).any():
                try:
                    valid_entries = df_copy[field].apply(lambda x: x if isinstance(x, dict) else {})
                    if not valid_entries.empty:
                        expanded = valid_entries.apply(pd.Series)
                        expanded.columns = [f"{field}_{col}" for col in expanded.columns]
                        df_copy = pd.concat([df_copy.drop(field, axis=1), expanded], axis=1)
                except Exception as e:
                    logging.warning(f"Could not expand nested field '{field}': {e}.")
            else:
                logging.debug(f"Column '{field}' has no dictionaries, skipping expansion.")
    return df_copy


def optimize_data_types(df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
    if df.empty: return df
    logging.info("Optimizing data types...")
    mem_usage_orig = df.memory_usage(deep=True).sum() / (1024**2)
    df_copy = df.copy()
    try:
        if date_column and date_column in df_copy.columns:
             df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
             if df_copy[date_column].isna().any():
                  logging.warning(f"NaNs introduced in '{date_column}' during datetime conversion.")

        for col in df_copy.select_dtypes(include=['object']).columns:
            if col == date_column: continue
            try:
                numeric_col = pd.to_numeric(df_copy[col], errors='coerce')
                if numeric_col.notna().sum() / df_copy[col].notna().sum() > 0.8:
                     df_copy[col] = numeric_col
                     logging.debug(f"Converted object column '{col}' to numeric.")
            except Exception: pass

        for col in df_copy.select_dtypes(include=np.number).columns:
            try:
                if pd.api.types.is_float_dtype(df_copy[col]):
                    df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
                elif pd.api.types.is_integer_dtype(df_copy[col]) and df_copy[col].notna().all():
                     df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
            except Exception: pass

        for col in df_copy.select_dtypes(include=['object']).columns:
             if col == date_column: continue
             try:
                 num_unique_values = df_copy[col].nunique()
                 if 1 < num_unique_values < 0.5 * len(df_copy):
                     df_copy[col] = df_copy[col].astype('category')
             except Exception: pass

        logging.info("Data type optimization complete.")
        mem_usage_new = df_copy.memory_usage(deep=True).sum() / (1024**2)
        logging.info(f"Memory usage reduced from {mem_usage_orig:.2f} MB to {mem_usage_new:.2f} MB")
        return df_copy

    except Exception as e:
        logging.error(f"Data type optimization failed: {e}", exc_info=True)
        return df


if __name__ == "__main__":
    end_date_dt = datetime.now()
    start_date_recent_dt = end_date_dt - timedelta(days=7)
    DATE_RANGE_RECENT = (start_date_recent_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))
    DATE_RANGE_HISTORICAL = ('2023-01-01', '2023-12-31')
    MAX_RECORDS_TRAFFIC = 200000
    MAX_RECORDS_EVENTS_RECENT = 50000
    MAX_RECORDS_EVENTS_HIST = 500000
    MAX_RECORDS_VOLUME = 500000 

    dataframes = {}

    try:
        logging.info(f"--- Fetching Recent Data ({DATE_RANGE_RECENT}) ---")
        dataframes['traffic'] = fetch_paginated_data('traffic', DATE_RANGE_RECENT, MAX_RECORDS_TRAFFIC)
        dataframes['events_current'] = fetch_paginated_data('events_current', DATE_RANGE_RECENT, MAX_RECORDS_EVENTS_RECENT)

        logging.info(f"--- Fetching Historical Data ({DATE_RANGE_HISTORICAL}) ---")
        dataframes['events_historical'] = fetch_paginated_data('events_historical', DATE_RANGE_HISTORICAL, MAX_RECORDS_EVENTS_HIST)
        dataframes['volume'] = fetch_paginated_data('volume', DATE_RANGE_HISTORICAL, MAX_RECORDS_VOLUME) 

        # Save DataFrames
        if 'traffic' in dataframes and not dataframes['traffic'].empty:
            dataframes['traffic'].to_parquet('traffic_data.parquet', engine='pyarrow', index=False)
            logging.info(f"Traffic data saved: {len(dataframes['traffic'])} records")

        events_list = []
        if 'events_current' in dataframes and not dataframes['events_current'].empty:
            events_list.append(dataframes['events_current'])
        if 'events_historical' in dataframes and not dataframes['events_historical'].empty:
             events_list.append(dataframes['events_historical'])

        if events_list:
             all_events_df = pd.concat(events_list, ignore_index=True)
             if 'event_id' in all_events_df.columns:
                  all_events_df.drop_duplicates(subset=['event_id'], inplace=True)
             all_events_df.to_parquet('events_all.parquet', engine='pyarrow', index=False)
             logging.info(f"Combined Events data saved: {len(all_events_df)} records")
        else:
             logging.warning("No event data fetched or saved.")

        if 'volume' in dataframes and not dataframes['volume'].empty:
            volume_df = dataframes['volume']
            logging.info(f"Processing fetched 'volume' data. Columns: {volume_df.columns.tolist()}")

            required_cols = ['yr', 'm', 'd', 'hh', 'mm']
            if all(col in volume_df.columns for col in required_cols):
                logging.info("Constructing timestamp column for volume data...")
                for col in required_cols:
                    volume_df[col] = pd.to_numeric(volume_df[col], errors='coerce')
                volume_df.dropna(subset=required_cols, inplace=True) 

                try:
                     date_str = (
                         volume_df['yr'].astype(int).astype(str) + '-' +
                         volume_df['m'].astype(int).astype(str).str.zfill(2) + '-' +
                         volume_df['d'].astype(int).astype(str).str.zfill(2) + ' ' +
                         volume_df['hh'].astype(int).astype(str).str.zfill(2) + ':' +
                         volume_df['mm'].astype(int).astype(str).str.zfill(2)
                     )
                     volume_df['timestamp'] = pd.to_datetime(date_str + ':00', format='%Y-%m-%d %H:%M:%S', errors='coerce')
                     volume_df.dropna(subset=['timestamp'], inplace=True) 
                     logging.info("Successfully constructed 'timestamp' column.")

                     start_hist_dt = pd.to_datetime(DATE_RANGE_HISTORICAL[0])
                     end_hist_dt = pd.to_datetime(DATE_RANGE_HISTORICAL[1])
                     volume_df_filtered = volume_df[
                         (volume_df['timestamp'] >= start_hist_dt) &
                         (volume_df['timestamp'] <= end_hist_dt)
                     ].copy()
                     logging.info(f"Filtered volume data locally to range {DATE_RANGE_HISTORICAL}. Records remaining: {len(volume_df_filtered)}")

                except Exception as e:
                    logging.error(f"Error constructing or filtering timestamp for volume data: {e}", exc_info=True)
                    volume_df_filtered = volume_df 
            else:
                logging.warning(f"Required columns {required_cols} not all present in volume data. Cannot construct timestamp or filter locally.")
                volume_df_filtered = volume_df

            if not volume_df_filtered.empty:
                 volume_df_filtered.to_parquet('volume_data.parquet', engine='pyarrow', index=False)
                 logging.info(f"Volume data saved: {len(volume_df_filtered)} records")
            else:
                 logging.warning("Volume data became empty after processing/filtering. Not saved.")
        else:
             logging.warning("No volume data fetched or saved.")

        logging.info("--- Data pipeline completed successfully ---")

    except KeyboardInterrupt:
        logging.warning("--- Pipeline interrupted by user ---")
    except Exception as e:
        logging.error(f"--- Critical failure in pipeline execution: {e} ---", exc_info=True)
