import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

  
def relu(x):
  return max(0, x)
  
def relu_derivative(x):
  return 0 if x < 0 else 1
  
class Digit_Recognition_ANN:
  
  def __init__(self, input_size, hidden_layers):
    self.input = input_size
    self.hidden_layers = hidden_layers
    self.output

  def forward(self, X):
    self.input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    self.output = relu(self.hidden_layers)
    
    self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.output = relu(self.output_input)
    
    return self.output
    
  def train(self, x, y, epochs, learning_rate): 
    
    

