import matplotlib.pyplot as plt
import numpy as np

class TestRegressionModel:
    def __init__(self, weights, X_test, y_test):
        self.coefficients = weights[1:]  # Exclude the intercept from weights
        self.intercept = weights[0]  # The first element is the intercept
        self.X_test = X_test
        self.y_test = y_test

    def predict(self):
        y_pred = []
        for i in range(len(self.X_test)):
            prediction = sum(self.coefficients[j] * self.X_test[i][j] for j in range(len(self.coefficients))) + self.intercept
            y_pred.append(prediction)
        return y_pred

    def test_model(self, method=0):
        y_pred = self.predict()

        if method == 0:
            # Calculate the Mean Squared Error
            mse = sum((self.y_test[i] - y_pred[i]) ** 2 for i in range(len(self.y_test))) / len(self.y_test)
            return mse
        elif method == 1:
            # Calculate the coefficient of determination (R^2)
            y_mean = sum(self.y_test) / len(self.y_test)
            ss_res = sum((self.y_test[i] - y_pred[i]) ** 2 for i in range(len(self.y_test)))
            ss_tot = sum((self.y_test[i] - y_mean) ** 2 for i in range(len(self.y_test)))
            r2 = 1 - (ss_res / ss_tot)
            return r2


  
    def plot_pred_and_actual(self):
        y_pred = self.predict()
        observations = range(len(self.y_test))
        
        # Plot only 50 percent of the values
        half_length = len(self.y_test) // 2
        observations = observations[:half_length]
        y_pred = y_pred[:half_length]
        self.y_test = self.y_test[:half_length]
        
        plt.figure(figsize=(12, 6))  # Make the figure larger in size
        plt.scatter(observations, y_pred, color='blue', label='Predicted')
        plt.scatter(observations, self.y_test, color='red', label='Actual')
        plt.plot(observations, y_pred, color='blue', linestyle='-', linewidth=0.5)
        plt.plot(observations, self.y_test, color='red', linestyle='-', linewidth=0.5)
        
        plt.xlabel('Observations')
        plt.ylabel('Values')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.show()