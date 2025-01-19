import pandas as pd
import matplotlib.pyplot as plt

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

def histogram(df):
    # Plot a histogram for 'age'
    plt.hist(df['age'], bins=20, color='blue', edgecolor='black')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # Plot a histogram for 'bmi'
    plt.hist(df['bmi'], bins=20, color='green', edgecolor='black')
    plt.title('Distribution of BMI')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.show()

    # Plot a histogram for 'charges'
    plt.hist(df['charges'], bins=20, color='orange', edgecolor='black')
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.show()


def box_plot(df):
    plt.figure(figsize=(12, 6))

    # Box plot for 'age'
    plt.figure(figsize=(6, 4))
    plt.boxplot(df['age'], vert=False, patch_artist=True, boxprops=dict(facecolor='blue'))
    plt.title('Box Plot of Age')
    plt.xlabel('Age')
    plt.show()

    # Box plot for 'bmi'
    plt.figure(figsize=(6, 4))
    plt.boxplot(df['bmi'], vert=False, patch_artist=True, boxprops=dict(facecolor='green'))
    plt.title('Box Plot of BMI')
    plt.xlabel('BMI')
    plt.show()

    # Box plot for 'charges'
    plt.figure(figsize=(6, 4))
    plt.boxplot(df['charges'], vert=False, patch_artist=True, boxprops=dict(facecolor='orange'))
    plt.title('Box Plot of Insurance Charges')
    plt.xlabel('Charges')
    plt.show()

    plt.tight_layout()
    plt.show()    


df = pd.read_csv('E:\\NumericalMethods\\Project\\DataSet.csv')

df_cleaned = clean_data(df)
# print(df_cleaned)
# print(len(df_cleaned))
histogram(df_cleaned)
