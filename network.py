import variable

class Network:
	def __init__(self):
		self.variables = []
		self.cells = []
		self.logForwardPropgation = False
		self.logBackwardPropagation = False
	def getVariablesAmount(self):
		return len(self.variables);
	def getCellsAmount(self):
		return len(self.cells);
	def appendVariable(self, variable):
		self.variables.append(variable);
	def appendCell(self, cell):
		self.cells.append(cell);
	def forwardPropagation(self):
		for cell in self.cells:
			if self.logForwardPropgation:
				print(cell.__class__.__name__)
			cell.forwardPropagation()
			if self.logForwardPropgation:
				cell.logOutputValue()
	def backwardPropagation(self):
		for cell in reversed(self.cells):
			if self.logBackwardPropagation:
				print(cell.__class__.__name__)
				cell.logOutputGradient()
			cell.backwardPropagation();
	def applyGradient(self, step_size):
		for variable in self.variables:
			variable.applyGradient(step_size);