import math
import numpy as np
import pandas as pd
from pre_process import preProcessor
from RegressionModel import SimpleRegressor
from data_gatherer import DataGatherer
from TestRegressionModel import TestRegressionModel

class StartPoint:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df_cleaned = None
        self.preprocessor = None
        self.simpleregressor = None
        self.coefficients_gradient = None
        self.coefficients_newton = None
        self.bmi_mean = None
        self.bmi_std = None
        self.risk_score_mean = None
        self.risk_score_std = None
        self.charges_mean = None
        self.charges_predicted_mean = None

    def preprocess_data(self):
        self.df_cleaned = DataGatherer.clean_data(self.df)
        self.preprocessor = preProcessor(self.df_cleaned)
        self.charges_mean = self.df_cleaned['charges'].mean()
        self.preprocessor.log_transform(self.df_cleaned)
        self.preprocessor.standardize(self.df_cleaned, 'bmi')
        self.preprocessor.create_composite_risk_score(self.df_cleaned)
        self.preprocessor.smoker_age_interaction(self.df_cleaned)
        self.preprocessor.family_size(self.df_cleaned)
        self.preprocessor.create_composite_bmi_age(self.df_cleaned)
        self.preprocessor.standardize(self.df_cleaned, 'risk_score')
        self.bmi_mean = self.df_cleaned['bmi'].mean()
        self.bmi_std = self.df_cleaned['bmi'].std()
        self.risk_score_mean = self.df_cleaned['risk_score'].mean()
        self.risk_score_std = self.df_cleaned['risk_score'].std()
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(self.df_cleaned)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.simpleregressor = SimpleRegressor(X_train.values.tolist(), y_train.values.tolist())
        self.coefficients_gradient = self.simpleregressor.gradient_descent()
        self.coefficients_newton = self.simpleregressor.newtons_method()

    def test_model(self, coefficients, X_test, y_test):
        test_model = TestRegressionModel(coefficients, X_test.values.tolist(), y_test.values.tolist())
        mse = test_model.test_model(0)
        test_model.plot_pred_and_actual()
        r2 = test_model.test_model(1)
        test_model.plot_pred_and_actual()
        return mse, r2

    def predict_charges(self, user_data, coefficients):
        intercept = coefficients[0]
        weights = coefficients[1:]
        prediction = intercept + sum(w * x for w, x in zip(weights, user_data))
        return prediction
    
    def predict_charges_with_scaling(self, user_data, coefficients):
        # Predict using risk_score and bmi
        risk_score = user_data['risk_score']
        bmi = user_data['bmi']
        prediction = self.predict_charges([risk_score, bmi], coefficients)
        
        # Scale prediction back to original scale
        scaled_prediction = math.exp(prediction)
        self.charges_predicted_mean = scaled_prediction.mean()
        



    def display_results(self, mse, r2, method):
        print(f"Mean Squared Error for {method}: {mse}")
        print(f"Coefficient of Determination for {method}: {r2}")

    def plot_contour(self):
        self.simpleregressor.plot_contour()

    def plot_convergence(self):
        self.simpleregressor.plot_convergence()

    def plot_cost_convergence(self):
        self.simpleregressor.plot_cost_convergence()

  

    def compute_and_display_measures(self):
        measures_age = DataGatherer.compute_measures(self.df_cleaned, 'age')
        measures_bmi = DataGatherer.compute_measures(self.df_cleaned, 'bmi')
        measures_charges = DataGatherer.compute_measures(self.df_cleaned, 'charges')

        # Create a DataFrame to display the results
        results_table = pd.DataFrame({
            'Variable': ['Age', 'BMI', 'Charges'],
            'Mean': [measures_age['Mean'], measures_bmi['Mean'], measures_charges['Mean']],
            'Median': [measures_age['Median'], measures_bmi['Median'], measures_charges['Median']],
            'Variance': [measures_age['Variance'], measures_bmi['Variance'], measures_charges['Variance']],
            'Standard Deviation': [measures_age['Standard Deviation'], measures_bmi['Standard Deviation'], measures_charges['Standard Deviation']]
        })

        print(results_table)

    def plot_correlation_matrix(self):
        DataGatherer.correlation_matrix(self.df_cleaned)

    def plot_scatter(self):
        self.preprocessor.plot_scatter(self.df_cleaned, 'risk_score', 'age', 'bmi', 'charges')


    def process_user_input(self, user_data):
        user_df = pd.DataFrame([user_data])

        # Create composite risk score
        user_df['risk_score'] = user_df['bmi'] * 0.1 + user_df['age'] * 0.3 + user_data['smoker'] * 0.6

        # Standardize BMI
        user_df['bmi'] = (user_df['bmi'] - self.bmi_mean) / self.bmi_std

        # Standardize risk score
        user_df['risk_score'] = (user_df['risk_score'] - self.risk_score_mean) / self.risk_score_std

        # Return processed user data with only 'bmi' and 'risk_score'
        return user_df[['risk_score', 'bmi']].values.tolist()[0]
        

    def change_to_original_scale(self, scaled_prediction):
        correction_factor = self.charges_mean / self.charges_predicted_mean

        adjusted_prediction = np.exp(scaled_prediction) * correction_factor
        return adjusted_prediction



def main():
    data_path = 'E:\\NumericalMethods\\Project\\DataSet.csv'
    start_point = StartPoint(data_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = start_point.preprocess_data()

    # Compute and display measures
    start_point.compute_and_display_measures()

    # Plot correlation matrix
    start_point.plot_correlation_matrix()

    # Plot scatter plots
    start_point.plot_scatter()

    # Train models
    start_point.train_model(X_train, y_train)

    # Print coefficients
    print("Coefficients achieved by Gradient Descent:", start_point.coefficients_gradient)
    print("Coefficients achieved by Newton's Method:", start_point.coefficients_newton)

    # Test Gradient Descent Model
    mse_gradient, r2_gradient = start_point.test_model(start_point.coefficients_gradient, X_test, y_test)
    start_point.display_results(mse_gradient, r2_gradient, "Gradient Descent")

    # Plotting for Gradient Descent
    start_point.plot_contour()
    start_point.plot_convergence()
    start_point.plot_cost_convergence()




    # Test Newton's Method Model
    mse_newton, r2_newton = start_point.test_model(start_point.coefficients_newton, X_test, y_test)
    start_point.display_results(mse_newton, r2_newton, "Newton's Method")

    # Plotting for Newton's Method
    start_point.plot_convergence()
    start_point.plot_cost_convergence()

    # Ask if the user wants to do predictions
    do_predict = input("Do you want to predict charges? (yes/no): ").lower()
    while do_predict not in ['yes', 'no']:
        do_predict = input("Invalid input. Do you want to predict charges? (yes/no): ").lower()

    if do_predict == 'yes':
        # Get user input
        age = float(input("Enter age: "))
        sex = input("Enter sex (male/female): ").lower()
        while sex not in ['male', 'female']:
            sex = input("Invalid input. Enter sex (male/female): ").lower()

        smoker = input("Are you a smoker? (yes/no): ").lower()
        while smoker not in ['yes', 'no']:
            smoker = input("Invalid input. Are you a smoker? (yes/no): ").lower()

        bmi = float(input("Enter BMI: "))
        children = int(input("Enter number of children (integer): "))
        
        region = input("Enter region (0: NorthWest, 1: SouthWest, 2: NorthEast, 3: SouthEast): ")
        while region not in ['0', '1', '2', '3']:
            region = input("Invalid input. Enter region (0: NorthWest, 1: SouthWest, 2: NorthEast, 3: SouthEast): ")
        region = int(region)

        # Encode user input
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0

        user_data = {
            'age': age,
            'sex': sex_encoded,
            'smoker': smoker_encoded,
            'bmi': bmi,
            'children': children,
            'region': region
        }

        # Process user input
        user_data = start_point.process_user_input(user_data)


        # Predict charges using both models
        charges_gradient = start_point.predict_charges(user_data, start_point.coefficients_gradient)
        charges_newton = start_point.predict_charges(user_data, start_point.coefficients_newton)

        print(f"Predicted charges using Gradient Descent: {start_point.change_to_original_scale(charges_gradient)}")
        print(f"Predicted charges using Newton's Method: {start_point.change_to_original_scale(charges_newton)}")

if __name__ == "__main__":
    main()