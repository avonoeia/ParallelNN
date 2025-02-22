import torch.nn as nn

def create_model(in_size=128, hidden_size=1000, num_layers=200, out_size=10, add_relu=True):
	"""
    Creates a neural network model with customizable input size, hidden size, number of layers, and output size.
	
	Args:
		in_size (int): Size of the input layer.
		hidden_size (int): Size of the hidden layers.
		num_layers (int): Number of hidden layers.
		out_size (int): Size of the output layer.
		add_relu (bool): Whether to add ReLU activation between layers.
	
	Returns:
		nn.Sequential: The created model.
	"""
	layers = [nn.Linear(in_size, hidden_size)]
	
	# hidden layers
	for _ in range(num_layers):
		layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
		if add_relu:
			layers.append(nn.ReLU())
	
	# output layer
	layers.append(nn.Linear(hidden_size, out_size))
	
	model = nn.Sequential(*layers)
	return model