from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class preProcessor():


    def __init__(self,df):
        
        self.label_encoding(df)
     
        

    
    def label_encoding(self, data):


        # Create a copy of the dataset to avoid modifying the original
        

        # Label encoding for 'sex'
        data['sex'] = data['sex'].map({'female': 0, 'male': 1})

        # Label encoding for 'smoker'
        data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

        # Label encoding for 'region'
        data['region'] = data['region'].map({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})

 
       

    def log_transform(self, data):
        # Log-transform 'charges'
        data['charges'] = np.log(data['charges'])
        # Display the updated dataset
        #print(data.head())

    def standardize(self, data,column = 'charges'):
        # Standardize the data
        data[column] = (data[column] - data[column].mean()) / data[column].std()
        # Display the first 5 rows of the standardized data
        print(data.head())    

    def create_composite_risk_score(self, data):
        # Create a composite risk score
        data['risk_score'] = data['age'] * 0.3 + data['bmi'] * 0.1 + data['smoker'] * 0.6
        # Display the updated dataset
        #print(data.head())

    
    def smoker_age_interaction(self, data):
        # Create a new feature 'smoker_age_interaction'
        data['smoker_age_interaction'] = data['smoker'] * data['age']
        # Display the updated dataset
        #print(data.head())

    def create_composite_bmi_age(self, data):
        # Create a composite risk score
        data['bmi_age'] = data['age'] * data['bmi'] 
        # Display the updated dataset
        #print(data.head())
    
    
    
    def family_size(self, data):
        # Create a new feature 'family_size'
        data['family_size'] = data['children'] + 1
        # Display the updated dataset
        #print(data.head())
   


    def split_data(self, data, target_variable='charges', test_size=0.2):
        # Split the data into training and test sets
        X = data[['risk_score', 'bmi']]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test




    def plot_scatter(self, data, col1, col2, col3, col4):
        """
        Plots scatter plots for every pair of the four specified columns.
        
        Parameters:
        data (DataFrame): The DataFrame containing the data.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        col3 (str): The name of the third column.
        col4 (str): The name of the fourth column.
        """
        plt.figure(figsize=(18, 12))

        # Plot col1 vs col2
        plt.subplot(2, 3, 1)
        plt.scatter(data[col1], data[col2], alpha=0.6)
        plt.title(f'{col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)

        # Plot col1 vs col3
        plt.subplot(2, 3, 2)
        plt.scatter(data[col1], data[col3], alpha=0.6)
        plt.title(f'{col1} vs {col3}')
        plt.xlabel(col1)
        plt.ylabel(col3)

        # Plot col1 vs col4
        plt.subplot(2, 3, 3)
        plt.scatter(data[col1], data[col4], alpha=0.6)
        plt.title(f'{col1} vs {col4}')
        plt.xlabel(col1)
        plt.ylabel(col4)

        # Plot col2 vs col3
        plt.subplot(2, 3, 4)
        plt.scatter(data[col2], data[col3], alpha=0.6)
        plt.title(f'{col2} vs {col3}')
        plt.xlabel(col2)
        plt.ylabel(col3)

        # Plot col2 vs col4
        plt.subplot(2, 3, 5)
        plt.scatter(data[col2], data[col4], alpha=0.6)
        plt.title(f'{col2} vs {col4}')
        plt.xlabel(col2)
        plt.ylabel(col4)

        # Plot col3 vs col4
        plt.subplot(2, 3, 6)
        plt.scatter(data[col3], data[col4], alpha=0.6)
        plt.title(f'{col3} vs {col4}')
        plt.xlabel(col3)
        plt.ylabel(col4)

        plt.tight_layout()
        plt.show()


    def plot_3d_scatter(self, data, target_variable='charges'):
        # 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(data['smoker'], data['risk_score'], data['bmi'], c=data[target_variable], cmap='viridis', alpha=0.6)
        ax.set_title('3D Scatter Plot')
        ax.set_xlabel('Smoker')
        ax.set_ylabel('Risk Score')
        ax.set_zlabel('BMI')
        fig.colorbar(sc, ax=ax, label=target_variable)
        plt.show()
      