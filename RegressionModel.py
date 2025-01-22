import numpy as np
import matplotlib.pyplot as plt



class SimpleRegressor:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.path_history = []  # Stores weight values during training
        self.X_train = None
        self.y_train = None
        self.feature_names = ['risk_score', 'bmi', 'smoker']

    def fit(self, X, y):
        # Initialize parameters
        num_features = len(X[0])
        self.weights = [0.0] * num_features
        self.bias = 0.0
        self.X_train = X
        self.y_train = y
        num_samples = len(X)
        self.weights_history = []  # Stores weight values during training
        self.bias_history = []
        # Store initial state
        self.path_history.append((self.weights.copy(), self.bias))
        
        # Training loop
        for _ in range(self.iterations):
            # Calculate predictions and error
            predictions = [sum(w*x for w,x in zip(self.weights, xi)) + self.bias for xi in X]
            errors = [p - yt for p, yt in zip(predictions, y)]
            
            # Update weights
            new_weights = [
                w - self.lr * sum(e*xi[i] for e,xi in zip(errors,X))/num_samples 
                for i,w in enumerate(self.weights)
            ]
            
            # Update bias
            new_bias = self.bias - self.lr * sum(errors)/num_samples
            
            # Store current parameters
            self.weights = new_weights
            self.bias = new_bias
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            self.path_history.append((self.weights.copy(), self.bias))
            
            # Track loss
            current_loss = sum(e**2 for e in errors)/num_samples
            self.loss_history.append(current_loss)

        return (self.weights, self.bias)

    def plot_contour(self, feature1=0, feature2=1):
        """Shows gradient descent path for 2 selected features"""
   
        
        # Create grid for selected features
        w1_vals = np.linspace(-1, 1, 50)
        w2_vals = np.linspace(-1, 1, 50)
        W1, W2 = np.meshgrid(w1_vals, w2_vals)
        
        # Calculate costs using fixed other parameters
        costs = np.zeros(W1.shape)
        final_bias = self.bias
        fixed_weight = self.weights[2]  # Weight for the third feature (smoker)
        
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                temp_weights = [W1[i,j], W2[i,j], fixed_weight]
                predictions = [
                    sum(w*x for w,x in zip(temp_weights, xi)) + final_bias 
                    for xi in self.X_train
                ]
                errors = [p - yt for p, yt in zip(predictions, self.y_train)]
                costs[i,j] = sum(e**2 for e in errors)/len(errors)
        
        # Plot contour
        plt.figure(figsize=(10, 6))
        plt.contour(W1, W2, costs, levels=np.logspace(-1, 3, 15), alpha=0.6)
        
        # Plot optimization path
        path = np.array([
            [step[0][feature1], step[0][feature2]] 
            for step in self.path_history
        ])
        plt.plot(path[:,0], path[:,1], 'r.-', markersize=5)
        plt.scatter(path[-1,0], path[-1,1], c='g', s=100, label='Final Weights')
        plt.title(f'Gradient Descent Path (Features {feature1+1} & {feature2+1})')
        plt.xlabel(f'Weight {feature1+1} (Risk Score)')
        plt.ylabel(f'Weight {feature2+1} (BMI)')
        plt.legend()
        plt.colorbar(label='Cost')
        plt.show()

    def plot_convergence(self):
        """Shows how the weights and bias change during training"""
        
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot weights convergence
        for i, weight_history in enumerate(zip(*self.weights_history)):
            axes[0].plot(weight_history, label=f'{self.feature_names[i]} (Weight {i+1})')
        axes[0].set_title('Weights Convergence')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Weight Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot bias convergence
        axes[1].plot(self.bias_history, label='Bias', color='black')
        axes[1].set_title('Bias Convergence')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Bias Value')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()