import numpy as np

# Back propagation
  # save activations and derivatives
  # implement back-propagation
  # implement gradient decent
  # implement train
  # train out net with some dummy dataset
  # make some predictions



class MLP(object):
  # A Multi Layer Perception Class.
  def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
    """
      Constructor for the MLP. 
      Takes in the number of inputs, 
      a variable number of hidden layers,
      and number of outputs.
      
      Args: 
        num_inputs (int): The number of inputs to the network.
        hidden_layers (list): A list of ints for the hidden layers.
        num_outputs (int): The number of outputs.
    """
    
    self.num_inputs = num_inputs
    self.hidden_layers = hidden_layers
    self.num_outputs = num_outputs
    
    # Create a generic representation of the layers
    layers = [num_inputs] + hidden_layers + [num_outputs]
    # print(layers)
    
   
    # Init weights
    weights = []
    for i in range(len(layers) - 1): # A three layer network has weights matrices between the layers ( 3-1 = 2)
      w = np.random.rand(layers[i], layers[i + 1])
      weights.append(w)
    self.weights = weights
    
    # Init activations
    activations = []
    for i in range(len(layers)):
      a = np.zeros(layers[i])
      activations.append(a)
    self.activations = activations
    
    # Init derivatives
    derivatives = []
    for i in range(len(layers) - 1):  # The derivatives with respect to the weights
      d = np.zeros((layers[i], layers[i + 1]))
      derivatives.append(d)
    self.derivatives = derivatives
    
    
  def forward_propagate(self, inputs):
    """
    Computes forward propagation of the network based on input signals. 
    
    Args:
      Inputs (ndarray): Input signals
    Returns:
      Activations (ndarray): Output values
    """
    activations = inputs
    self.activations[0] = inputs
    
    for i, w in enumerate(self.weights):
      # calculate matrix multiplication between previous activation and weight matrix
      net_inputs = np.dot(activations, w)
      
      # apply sigmoid activation function
      activations = self._sigmoid(net_inputs)
      self.activations[i + 1] = activations
      
    return activations
  
  
  
  def back_propagate( self, error, verbose=False):
    
    """
      We want to calculate the derivative of the error width respect to the weight.
        Formula (Matrices multiplication): dE / dW_1 = (y - a_[i+1]) s`(h_[i+1]) a_i
          - (y - a_[i+1]) = error(actual value - the prediction) comes from the ARG
          - s`(h_[i+1]) = derivative of the sigmoid function
          - a_i = activations calculated at index i
          
      The next we want to calculate is the sigma prime (s`(h_[i+1])):
        Formula: s`(h_[i+1]) = s(h_[i+1]) (1 - s(h_[i+1])) (sigma prime is the sigma itself multiplied with (1 - the sigma))
      
      The sigma calculated at index i+1 (s(h_[i+1])) is basically the activation:
        Formula: (s(h_[i+1])) = a_[i+1]
        
        
      Calculate the next derivative towards the left (right --> left)
      Formula: dE / dW_[i-1] = (y - a_[i+1]) s`(h_[i+1]) W_i s`(h_i) a_[i-1]
        - (y - a_[i+1]) s`(h_[i+1]) = previous derivative (delta)
        - W_i s`(h_i) a_[i-1]
      
    """
  
  
    for i in reversed(range(len(self.derivatives))): # With normal incremental of i, reversed makes it go right to left. 
      activations = self.activations[i + 1] # a_[i+1]
      
      # get derivative of the error with respect to w_i (dE / dW_1)
      delta = error * self._sigmoid_derivative(activations) # (y - a_[i+1]) s`(h_[i+1]) ---> ndarray([0.1, 0.2]) ---> ndarray([[0.1, 0.2]])
      delta_reshaped = delta.reshape(delta.shape[0], -1).T
      
      # get current activations (a_i)
      current_activations = self.activations[i] #  ---> ndarray([0.1, 0.2]) ---> ndarray([[0.1],[0.2]])
      current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
      
      
      self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped) # dE / dW_1 = (y - a_[i+1]) s`(h_[i+1]) a_i
      
      error = np.dot(delta, self.weights[i].T) # dE / dW_[i-1] = (y - a_[i+1]) s`(h_[i+1]) W_i
    
      if verbose:
        print(f"Layer {i}: Derivatives:\n{self.derivatives[i]}")
    
    
    return error  
      
        
  def _sigmoid_derivative(self, x):
    return x * (1.0 - x)
  
  
  def _sigmoid(self, x):
    """
    Sigmoid activation function
    
    Args:
      x (float): Value to be precessed
    Returns:
      y (float): Output  
    """
    y = 1.0 / (1 + np.exp(-x))
    return y
    
  
if __name__ == "__main__":
  
  # Create an instance of the MLP class
  mlp = MLP(2, [5], 1)
  
  # Create dummy data
  input = np.array([0.1, 0.2]) 
  target = np.array([0.3])
  
  # Forward propagation
  output = mlp.forward_propagate(input)
  
  # calculate the error
  error = target - output
  
  # Back propagation
  mlp.back_propagate(error, verbose=True)
  
      


