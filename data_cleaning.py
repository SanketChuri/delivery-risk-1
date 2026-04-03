import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def inspect_data(df):
    print("\n=== FIRST 5 ROWS ===\n")
    print(df.head())

    print("\n=== COLUMN NAMES ===\n")
    print(df.columns.tolist())

    print("\n=== DATA TYPES ===\n")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===\n")
    print(df.isnull().sum())

    print("\n=== DUPLICATES ===\n")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n=== UNIQUE VALUES ===\n")
    for col in df.select_dtypes(include='object').columns:
        print(f"\n{col}: {df[col].unique()}")


def clean_data(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    df['priority'] = df['priority'].str.lower()
    df['status'] = df['status'].str.lower()
    df['traffic_level'] = df['traffic_level'].str.lower()

    df['scheduled_time'] = pd.to_numeric(df['scheduled_time'], errors='coerce')
    df['actual_time'] = pd.to_numeric(df['actual_time'], errors='coerce')

    df['actual_time'] = df['actual_time'].fillna(df['scheduled_time'])
    df = df.dropna(subset=['scheduled_time'])

    df = df.drop_duplicates()

    return df