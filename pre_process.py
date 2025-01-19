import pandas as pd

class preProcessor():


    def __init__(self,df):
        self.df_copy = df.copy()
        self.label_encoding(self.df_copy)

    
    def label_encoding(self, data):


        # Create a copy of the dataset to avoid modifying the original
        

        # Label encoding for 'sex'
        data['sex'] = data['sex'].map({'female': 0, 'male': 1})

        # Label encoding for 'smoker'
        data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})

        # Label encoding for 'region'
        data['region'] = data['region'].map({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})

        # Display the updated dataset
        #print(df_numeric.head())
        print(data.head())
