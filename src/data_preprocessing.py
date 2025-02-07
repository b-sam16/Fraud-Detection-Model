import pandas as pd
import numpy as np

def handle_missing_values(df: pd.DataFrame, strategy='drop', fill_value=None) -> pd.DataFrame:
    """
    Handles missing values by either dropping them or filling them with a specified value.
    :param df: DataFrame to process
    :param strategy: 'drop' to remove missing values, 'fill' to replace them
    :param fill_value: Value to use when filling missing data (if strategy='fill')
    :return: Processed DataFrame
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill' and fill_value is not None:
        df = df.fillna(fill_value)
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the dataset.
    :param df: DataFrame to process
    :return: DataFrame with duplicates removed
    """
    return df.drop_duplicates()

def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects data types for specific columns.
    - Converts timestamps to datetime
    - Ensures categorical variables are of type 'category'
    :param df: DataFrame to process
    :return: DataFrame with corrected data types
    """
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['age'] = df['age'].astype(int)
    df['source'] = df['source'].astype('category')
    df['browser'] = df['browser'].astype('category')
    df['sex'] = df['sex'].astype('category')
    return df

def ip_to_integer(ip: str) -> int:
    """
    Converts an IP address to an integer format.
    :param ip: IP address as string
    :return: Integer representation of IP address
    """
    try:
        parts = list(map(int, ip.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except:
        return np.nan  # Return NaN for invalid IPs

def merge_with_ip(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges fraud dataset with IP-to-country mapping based on IP address.
    The IP address in fraud_df is converted to an integer and mapped to a country.
    :param fraud_df: Fraud transaction dataset
    :param ip_df: IP address mapping dataset
    :return: Merged DataFrame
    """
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(ip_to_integer)
    
    # Merging: Find the country for the corresponding IP range
    merged_df = fraud_df.merge(ip_df, how='left', 
                               left_on='ip_address_int', 
                               right_on='lower_bound_ip_address')
    
    # Drop unnecessary columns
    merged_df.drop(['ip_address_int', 'lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
    
    return merged_df
