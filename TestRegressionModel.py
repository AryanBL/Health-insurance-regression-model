import matplotlib.pyplot as plt

class TestRegressionModel:
    def __init__(self, coefficients, intercept, X_test, y_test):
        self.coefficients = coefficients
        self.intercept = intercept
        self.X_test = X_test
        self.y_test = y_test

    def predict(self):
        y_pred = []
        for i in range(len(self.X_test)):
            y_pred.append(sum(self.coefficients[j] * self.X_test[i][j] for j in range(len(self.X_test[i]))) + self.intercept)
        return y_pred    

    def test_model(self,errorComputer = 0):
        y_pred = self.predict()

        match errorComputer:
            case 0:
                # Calculate the Mean Squared Error
                mse = sum((yt - yp) ** 2 for yt, yp in zip(self.y_test, y_pred)) / len(self.y_test)
                return mse
            


  
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