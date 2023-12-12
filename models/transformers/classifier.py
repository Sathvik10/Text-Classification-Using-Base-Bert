import torch
import torch.nn as nn

class FNN1LayerClassifier(nn.Module):
	def __init__(self, num_classes):
		super(FNN1LayerClassifier, self).__init__()
		self.dropout1 = nn.Dropout(0.1)  # Dropout layer if needed
		self.linear = nn.Linear(768, num_classes)  # Example: Connect BERT output (768) to a hidden layer (128)
		self.act1 = nn.Softmax()

	def forward(self, input):
		x = self.dropout1(input)
		x = self.act1(self.linear(x))
		return x


class FNN2LayerClassifier(nn.Module):
	def __init__(self, num_classes):
		super(FNN2LayerClassifier, self).__init__()
		self.dropout1 = nn.Dropout(0.1)  # Dropout layer if needed
		self.linear = nn.Linear(768, 128)  # Connect BERT output (768) to a hidden layer (128)
		self.act1 = nn.ReLU()
		self.batch_norm = nn.LayerNorm(128)
		self.dropout2 = nn.Dropout(0.1)
		self.out = nn.Linear(128, num_classes)
		self.act2 = nn.Softmax()
	  

	def forward(self, input):
		x = self.dropout1(input)
		x = self.act1(self.linear(x))
		x = self.batch_norm(x)
		x = self.dropout2(x)
		x = self.act2(self.out(x))
		return x