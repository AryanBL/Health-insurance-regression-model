import pandas as pd

def clean_data(df):
    # Remove rows with missing values
    df = df.dropna()

    # Detect and remove outliers using IQR
    for col in ['age', 'bmi', 'charges']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Debug print statements
        print(f"Column: {col}")
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        print(f"Rows before filtering: {len(df)}")

        # Filter out outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Debug print statement
        print(f"Rows after filtering: {len(df)}")
    df = df.reset_index(drop=True)
    return df

df = pd.read_csv('E:\\NumericalMethods\\Project\\DataSet.csv')

df_cleaned = clean_data(df)
print(df_cleaned)
print(len(df_cleaned))