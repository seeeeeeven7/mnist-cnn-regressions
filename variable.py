import numpy as np

class Variable:
	def __init__(self, value = None):
		if value is not None:
			self.value = value
			self.gradient = np.zeros_like(value, dtype=np.float)
		else:
			self.gradient = None
	def __str__(self):
		return '{ value =\n' + str(self.value) + ',\n gradient =\n' + str(self.gradient) + '}';
	def __repr__(self):
		return self.__str__();
	def getOutput(self):
		return self
	def takeInput(self, value):
		self.value = np.reshape(value, self.value.shape);
		self.gradient = np.zeros_like(self.value, dtype=np.float)
	def applyGradient(self, step_size):
		self.value = self.value + self.gradient * step_size
		self.gradient = np.zeros_like(self.gradient, dtype=np.float)
	@staticmethod
	def random():
		return Variable(random.random() * 2 - 1);