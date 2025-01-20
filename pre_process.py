import numpy as np
import pandas as pd

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
        data['risk_score'] = data['age'] * 0.1 + data['bmi'] * 0.3 + data['smoker'] * 0.6
        # Display the updated dataset
        #print(data.head())

    def family_size(self, data):
        # Create a new feature 'family_size'
        data['family_size'] = data['children'] + 1
        # Display the updated dataset
        #print(data.head())