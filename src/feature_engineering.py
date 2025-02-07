import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature: transaction frequency for each user.
    """
    df['transaction_frequency'] = df.groupby('user_id')['purchase_value'].transform('count')
    return df

def transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature: transaction velocity (rate of transactions per day).
    """
    df['days_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.days + 1
    df['transaction_velocity'] = df['transaction_frequency'] / df['days_since_signup']
    return df

def time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based features: hour_of_day, day_of_week.
    """
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df

def scale_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Scales selected columns using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encodes categorical columns into numeric labels.
    """
    encoder = LabelEncoder()
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df
