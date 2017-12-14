from mnist import MNIST
import numpy as np
import random

# 数据准备

mndata = MNIST('data')
images, labels = mndata.load_training()
images_test, labels_test = mndata.load_testing()

images = np.array(images)
labels = np.array(labels)
images_test = np.array(images_test)
labels_test = np.array(labels_test)

images = images / 255
images_test = images_test / 255

def toOneHot(labels):
	labels_new = np.zeros([len(labels), 10])
	for index in range(len(labels)):
		labels_new[index][labels[index]] = 1
	return labels_new

labels = toOneHot(labels)
labels_test = toOneHot(labels_test)

# 输入层

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
		self.value = value;
		self.gradient = np.zeros_like(value, dtype=np.float)
	def applyGradient(self, step_size):
		self.value = self.value + self.gradient * step_size
		self.gradient = np.zeros_like(self.gradient, dtype=np.float)
	@staticmethod
	def random():
		return Variable(random.random() * 2 - 1);

class Cell:
	def getOutput(self):
		return self.output

class SCell(Cell):
	def __init__(self, input):
		self.input = input

class DCell(Cell):
	def __init__(self, input0, input1):
		self.input0 = input0
		self.input1 = input1

class AddCell(DCell):
	def forwardPropagation(self):
		self.output = Variable(self.input0.getOutput().value + self.input1.getOutput().value)
	def backwardPropagation(self):
		self.input0.getOutput().gradient += self.output.gradient
		self.input1.getOutput().gradient += self.output.gradient

class MatMulCell(DCell):
	def forwardPropagation(self):
		self.output = Variable(np.dot(self.input0.getOutput().value, self.input1.getOutput().value))
	def backwardPropagation(self):
		self.input0.getOutput().gradient += np.dot(self.output.gradient, self.input1.getOutput().value.T)
		self.input1.getOutput().gradient += np.dot(self.input0.getOutput().value.T, self.output.gradient)

class SoftmaxCell(SCell):
	def forwardPropagation(self):
		self.output = Variable(np.exp(self.input.getOutput().value) / np.sum(np.exp(self.input.getOutput().value)))
	def backwardPropagation(self):
		self.input.getOutput().gradient += -np.sum(self.output.gradient * self.output.value) * self.output.value + self.output.gradient * self.output.value;

class CrossEntropyCell(DCell):
	def forwardPropagation(self):
		self.output = Variable(-np.sum(np.log(self.input0.getOutput().value) * self.input1.getOutput().value))
	def backwardPropagation(self):
		self.input0.getOutput().gradient += self.output.gradient * -self.input1.getOutput().value / self.input0.getOutput().value
		self.input1.getOutput().gradient += self.output.gradient * -np.log(self.input0.getOutput().value)

class Convolution2DCell(Cell):
	def __init__(self, input, core):
		self.input = input
		self.core = core
	def forwardPropagation(self):
		

class Network:
	def __init__(self):
		self.variables = [];
		self.cells = [];
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
			cell.forwardPropagation();
	def backwardPropagation(self):
		for cell in reversed(self.cells):
			cell.backwardPropagation();
	def applyGradient(self, step_size):
		for variable in self.variables:
			variable.applyGradient(step_size);

# Network
network = Network();

# Variables
W = Variable(np.zeros([784, 10]))
network.appendVariable(W);

B = Variable(np.zeros([1, 10]))
network.appendVariable(B);

# Inputs Layer
X = Variable(np.zeros([1, 784]))
Y = Variable(np.zeros([1, 10]))

# Weighted-Sum
matmulCell = MatMulCell(X, W) # X * W => [10, 1]
network.appendCell(matmulCell)

# Bias
addCell = AddCell(matmulCell, B) # X * W + B => [10, 1]
network.appendCell(addCell)

# Softmax
softmaxCell = SoftmaxCell(addCell) # Softmax(X * W + B) => [10, 1]
network.appendCell(softmaxCell)

# Loss (Cross-entropy)
loss = CrossEntropyCell(softmaxCell, Y) # CrossEntropy(Softmax(X * W + B), Y) => Loss
network.appendCell(loss)

# Training
BATCH_NUMBER = 1000 # BATCH的数量
BATCH_SIZE = 100 # BATCH的大小
LEARNING_RATE = 0.5 #学习速率
for batch_index in range(BATCH_NUMBER ):
    # 构造一个BATCH
    batch_xs = [];
    batch_ys = [];
    for data_index in range(BATCH_SIZE):
        j = random.randint(0, len(images) - 1)
        x = images[j][np.newaxis];
        y = labels[j][np.newaxis];
        batch_xs.append(x);
        batch_ys.append(y);

    # 使用这个BATCH进行训练
    batch_loss = 0
    for data_index in range(BATCH_SIZE):
        x = batch_xs[data_index];
        y = batch_ys[data_index];
        X.takeInput(x);
        Y.takeInput(y);
        network.forwardPropagation() # 正向传播
        batch_loss += loss.getOutput().value # 统计整个BATCH的损失
        loss.getOutput().gradient = -1 / BATCH_SIZE # 整个BATCH统一计算梯度，所以单个数据点的输出梯度只有1/BATCH_SIZE
        network.backwardPropagation() # 反向传播
    
    # 引用梯度
    network.applyGradient(LEARNING_RATE)
    print('batch', batch_index, ', loss =', batch_loss)

# Test
precision = 0
for index in range(len(images_test)):
    x = images_test[index][np.newaxis];
    y = labels_test[index][np.newaxis];
    X.takeInput(x);
    Y.takeInput(y);
    network.forwardPropagation()
    predict = np.argmax(softmaxCell.getOutput().value)
    if predict == np.argmax(y):
        precision += 1 / len(images_test)
print('Precision =', precision)