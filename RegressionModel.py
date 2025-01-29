import numpy as np
import matplotlib.pyplot as plt



class SimpleRegressor:
    def __init__(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        self.cost_history = []
        self.path_history = []  # Stores weight values during training
        self.feature_names = ['risk_score', 'bmi']

    def gradient_descent(self , learning_rate=0.01, tolerance=1e-6, max_iterations=1000, decay_factor=1):
       
        X = self.X_train
        y = self.y_train
        
        # Number of samples (m) and features (n)
        m = len(X)
        n = len(X[0])
        
        # Add a column of ones to X for the bias term (intercept)
        X = [[1] + row for row in X]  # Adding bias (X becomes m x (n+1))
        
        # Initialize weights (n+1 x 1)
        weights = [0] * (n + 1)
        
        # Initialize cost history and previous cost, and path history
        self.path_history = [weights[:]]
        prev_cost = float('inf')
        
        for iteration in range(max_iterations):
            # Step 1: Compute predictions
            predictions = [sum(weights[j] * X[i][j] for j in range(n + 1)) for i in range(m)]
            
            # Step 2: Compute errors
            errors = [predictions[i] - y[i] for i in range(m)]
            
            # Step 3: Compute cost (mean squared error)
            cost = sum(e**2 for e in errors) / (2 * m)
            self.cost_history.append(cost)
            
            # Step 4: Check for convergence
            if abs(prev_cost - cost) < tolerance:
                print(f"Converged after {iteration} iterations.")
                break
            prev_cost = cost
            
            # Step 5: Compute gradient
            gradients = [0] * (n + 1)  # Gradient for each weight
            for j in range(n + 1):  # For each weight
                gradients[j] = sum(errors[i] * X[i][j] for i in range(m)) / m
            
            # Step 6: Update weights
            for j in range(n + 1):
                weights[j] -= learning_rate * gradients[j]

            self.path_history.append(weights[:])

            # Step 7: Update learning rate
            learning_rate *= decay_factor    
        
        return weights
    

    def newtons_method(self, tolerance=1e-6, max_iterations=100):
        
        self.cost_history.clear()
        X = self.X_train
        y = self.y_train

        # Number of samples (m) and features (n)
        m = len(X)
        n = len(X[0])

        # Add a column of ones to X for the bias term (intercept)
        X = [[1] + row for row in X]  # Adding bias (X becomes m x (n+1))

        # Initialize weights (n+1 x 1)
        weights = [0] * (n + 1)

        # Initialize cost history and path history
        self.path_history = [weights[:]]

        for iteration in range(max_iterations):
            # Step 1: Compute predictions
            predictions = [sum(weights[j] * X[i][j] for j in range(n + 1)) for i in range(m)]

            # Step 2: Compute errors
            errors = [predictions[i] - y[i] for i in range(m)]

            # Step 3: Compute cost (mean squared error)
            cost = sum(e**2 for e in errors) / (2 * m)
            self.cost_history.append(cost)

            # Step 4: Compute gradient
            gradients = [0] * (n + 1)
            for j in range(n + 1):
                gradients[j] = sum(errors[i] * X[i][j] for i in range(m)) / m

            # Step 5: Compute Hessian
            hessian = [[0] * (n + 1) for _ in range(n + 1)]
            for j in range(n + 1):
                for k in range(n + 1):
                    hessian[j][k] = sum(X[i][j] * X[i][k] for i in range(m)) / m

            # Step 6: Solve for weight updates using Newton's method formula
            hessian_inv = np.linalg.inv(hessian)

            weight_update = [0] * (n + 1)
            for i in range(n + 1):
                weight_update[i] = sum(hessian_inv[i][j] * gradients[j] for j in range(n + 1))

            # Update weights
            weights = [weights[j] - weight_update[j] for j in range(n + 1)]

            self.path_history.append(weights[:])

            # Check for convergence
            if max(abs(w) for w in weight_update) < tolerance:
                print(f"Converged after {iteration} iterations.")
                break

        return weights




    def compute_cost(self, w0, w1, w2):
        X = self.X_train
        y = self.y_train
        m = len(y)
        cost = 0
        for i in range(m):
            prediction = w0 + w1 * X[i][0] + w2 * X[i][1]
            error = prediction - y[i]
            cost += error ** 2
        return cost / (2 * m)

    
    def plot_contour(self):
        w0 = 0  # Keep w0 fixed
        w1_range = np.linspace(-5, 5, 100)  # Define a range for w1
        w2_range = np.linspace(-5, 5, 100)  # Define a range for w2

        # Create a grid of w1 and w2 values
        W1, W2 = np.meshgrid(w1_range, w2_range)
        Z = np.zeros_like(W1)

        # Compute the cost for each combination of w1 and w2
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                Z[i, j] = self.compute_cost(w0, W1[i, j], W2[i, j])

        # Create the contour plot with lines
        plt.figure(figsize=(10, 8))
        contour = plt.contour(W1, W2, Z, levels=50, cmap="viridis")
        plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

        # Extract path history for w1 and w2
        path_w1 = [point[1] for point in self.path_history]
        path_w2 = [point[2] for point in self.path_history]

        # Plot the path history
        plt.plot(path_w1, path_w2, marker="o", color="red", label="Path History")
        plt.scatter(path_w1[-1], path_w2[-1], color="black", label="Final Point", zorder=5)

        # Add titles and labels
        plt.title("Contour Plot with Path History")
        plt.xlabel("risk_score")
        plt.ylabel("bmi")
        plt.legend()
        plt.show()




        
    def plot_convergence(self):
        
        
        # Create subplots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot weights convergence
        for i in range(len(self.feature_names) + 1):
            weight_history = [weights[i] for weights in self.path_history]
            label = 'Bias' if i == 0 else f'{self.feature_names[i-1]} (Weight {i})'
            ax.plot(weight_history, label=label)
        
        ax.set_title('Weights Convergence')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight Value')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()


    def plot_cost_convergence(self):
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, label="Cost Function", color="blue")
        plt.title("Cost Function Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.show()
