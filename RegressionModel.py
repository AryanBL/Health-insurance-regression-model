class RegressionModel:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def gauss_elimination(self, matrix, vector):
        """
        Solve a system of linear equations using Gauss Elimination.
        Args:
            matrix (list of lists): Coefficient matrix.
            vector (list): Right-hand side vector.
        Returns:
            list: Solution vector.
        """
        n = len(matrix)

        # Forward elimination
        for i in range(n):
            # Partial pivoting to improve numerical stability
            max_row = i
            for k in range(i + 1, n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k
            
            # Swap rows
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            vector[i], vector[max_row] = vector[max_row], vector[i]

            # Make all rows below this one 0 in the current column
            for k in range(i + 1, n):
                factor = matrix[k][i] / matrix[i][i]
                for j in range(i, n):
                    matrix[k][j] -= factor * matrix[i][j]
                vector[k] -= factor * vector[i]

        # Back substitution
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = vector[i] / matrix[i][i]
            for k in range(i - 1, -1, -1):
                vector[k] -= matrix[k][i] * x[i]

        return x

    def linear_regression_gauss(self):
        """
        Perform linear regression using Gauss Elimination.
        Args:
            X (list of lists): Feature matrix (each row is a data point, each column is a feature).
            y (list): Output vector.
        Returns:
            list: Coefficients of the linear regression model.
        """

        X = self.X_train
        y = self.y_train
        # Add bias term (column of 1s) to X
        n_samples = len(X)
        n_features = len(X[0])
        X_bias = [[1] + row for row in X]

        # Print X_bias matrix
        # print("X_bias matrix:")
        # for row in X_bias:
        #     print(row)

        # Calculate normal equation components: X^T * X and X^T * y
        A = [[0] * (n_features + 1) for _ in range(n_features + 1)]
        b = [0] * (n_features + 1)

        # Fill in A and b matrices
        for i in range(n_features + 1):
            for j in range(n_features + 1):
                if i == 0 and j == 0:
                    A[i][j] = 1  # First element is the number of samples
                else:
                    A[i][j] = sum(X_bias[k][i] * X_bias[k][j] for k in range(n_samples))/n_samples
            b[i] = sum(X_bias[k][i] * y[k] for k in range(n_samples))/n_samples

        # Solve the normal equation using Gauss Elimination
        coefficients = self.gauss_elimination(A, b)
        # print("b matrix:")
        # for row in b:
        #     print(row)
        return coefficients
    

    def gradient_descent(self ,learning_rate=0.1, iterations=1000):
        """
        Perform gradient descent to optimize linear regression coefficients.
        Args:
            X (list of lists): Feature matrix (including bias term).
            y (list): Target vector.
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations.
        Returns:
            tuple: Optimized coefficients and history of coefficients.
        """

        X = self.X_train
        y = self.y_train

        m = len(y)  # Number of observations
        n = len(X[0])  # Number of features (including bias)

        # Initialize coefficients to zero
        coefficients = [0] * n
        coefficients_history = [[] for _ in range(n)]  # Store history for each coefficient

        for _ in range(iterations):
            # Calculate predictions
            predictions = [sum(coefficients[j] * X[i][j] for j in range(n)) for i in range(m)]

            # Compute gradients
            gradients = [0] * n
            for j in range(n):
                gradients[j] = sum((predictions[i] - y[i]) * X[i][j] for i in range(m)) / m

            # Update coefficients
            for j in range(n):
                coefficients[j] -= learning_rate * gradients[j]
                coefficients_history[j].append(coefficients[j])  # Track history

        return coefficients

