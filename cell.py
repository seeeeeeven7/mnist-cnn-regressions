import variable as var
import numpy as np
import scipy.signal

# 配置
np.set_printoptions(threshold=np.nan)
eps = 1e-6

# 基础类

class Cell:
	def getOutput(self):
		return self.output
	def logOutputValue(self):
		print(self.output.value)
	def logOutputGradient(self):
		print(self.output.gradient)

class SCell(Cell):
	def __init__(self, input):
		self.input = input

class DCell(Cell):
	def __init__(self, input0, input1):
		self.input0 = input0
		self.input1 = input1

# 计算单元类

class AddCell(DCell):
	def forwardPropagation(self):
		self.output = var.Variable(self.input0.getOutput().value + self.input1.getOutput().value)
	def backwardPropagation(self):
		self.input0.getOutput().gradient += self.output.gradient
		self.input1.getOutput().gradient += self.output.gradient
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class MatMulCell(DCell):
	def forwardPropagation(self):
		self.output = var.Variable(np.dot(self.input0.getOutput().value, self.input1.getOutput().value))
	def backwardPropagation(self):
		self.input0.getOutput().gradient += np.dot(self.output.gradient, self.input1.getOutput().value.T)
		self.input1.getOutput().gradient += np.dot(self.input0.getOutput().value.T, self.output.gradient)
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class ConvolutionCell(Cell):
	def __init__(self, image, core):
		self.image = image
		self.core = core
	def forwardPropagation(self):
		[n, m, c1] = self.image.getOutput().value.shape
		[p, q, c1, c2] = self.core.getOutput().value.shape
		self.output = var.Variable(np.zeros([n, m, c2]));
		for k1 in range(c1):
			for k2 in range(c2):
				self.output.value[:,:,k2] += scipy.signal.convolve2d(self.image.getOutput().value[:,:,k1], self.core.getOutput().value[:,:,k1,k2], mode = 'same', boundary = 'fill', fillvalue = 0)
	def backwardPropagation(self):
		[n, m, c1] = self.image.getOutput().value.shape
		[p, q, c1, c2] = self.core.getOutput().value.shape
		for k1 in range(c1):
			for k2 in range(c2):
				# 计算image的偏导
				flip = np.flipud(np.fliplr(self.core.getOutput().value[:,:,k1,k2]))
				self.image.getOutput().gradient[:,:,k1] += scipy.signal.convolve2d(self.output.gradient[:,:,k2], flip, mode = 'same', boundary = 'fill', fillvalue = 0)
				# 计算core的偏导
				pad = np.lib.pad(self.image.getOutput().value[:,:,k1], [(p // 2, q // 2), (p // 2, q // 2)], 'constant')
				self.core.getOutput().gradient[:,:,k1,k2] += scipy.signal.convolve2d(pad, self.output.gradient[:,:,k2], mode = 'valid')
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class ReLuCellType1(Cell):
	def __init__(self, input, bias):
		self.input = input
		self.bias = bias
	def forwardPropagation(self):
		[n, m, c] = self.input.getOutput().value.shape
		self.output = var.Variable(np.zeros([n, m, c]))
		for i in range(n):
			for j in range(m):
				for k in range(c):
					self.output.value[i][j][k] = max(self.input.getOutput().value[i][j][k] + self.bias.getOutput().value[k], 0)
	def backwardPropagation(self):
		[n, m, c] = self.input.getOutput().value.shape
		for i in range(n):
			for j in range(m):
				for k in range(c):
					if self.output.value[i][j][k] > eps: # 仅传递激活了的神经元
						self.input.getOutput().gradient[i][j][k] += self.output.gradient[i][j][k]
						self.bias.getOutput().gradient[k] += self.output.gradient[i][j][k]
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class ReLuCellType2(Cell):
	def __init__(self, input, bias):
		self.input = input
		self.bias = bias
	def forwardPropagation(self):
		[n, m] = self.input.getOutput().value.shape
		self.output = var.Variable(np.zeros([n, m]))
		for i in range(n):
			for j in range(m):
				self.output.value[i][j] = max(self.input.getOutput().value[i][j] + self.bias.getOutput().value[j], 0)
	def backwardPropagation(self):
		[n, m] = self.input.getOutput().value.shape
		for i in range(n):
			for j in range(m):
				if self.output.value[i][j] > eps: # 仅传递激活了的神经元
					self.input.getOutput().gradient[i][j] += self.output.gradient[i][j]
					self.bias.getOutput().gradient[j] += self.output.gradient[i][j]
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class MaxPoolingCell(Cell):
	def __init__(self, input, stepSize):
		self.input = input
		self.stepSize = stepSize
	def forwardPropagation(self):
		[n, m, c] = self.input.getOutput().value.shape
		[p, q] = self.stepSize
		self.output = var.Variable(np.zeros([(n + p - 1) // p, (m + q - 1) // q, c]))
		for i in range(n):
			for j in range(m):
				for k in range(c):
					self.output.value[i // p][j // q][k] = max(self.output.value[i // p][j // q][k], self.input.getOutput().value[i][j][k])
	def backwardPropagation(self):
		[n, m, c] = self.input.getOutput().value.shape
		[p, q] = self.stepSize
		for i in range(n):
			for j in range(m):
				for k in range(c):
					if self.output.value[i // p][j // q][k] > self.input.getOutput().value[i][j][k] - eps:
						self.input.getOutput().gradient[i][j][k] += self.output.gradient[i // p][j // q][k]
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class ReShapeCell(Cell):
	def __init__(self, input, shape):
		self.input = input
		self.shape = shape
	def forwardPropagation(self):
		self.output = var.Variable(np.reshape(self.input.getOutput().value, self.shape))
	def backwardPropagation(self):
		self.input.getOutput().gradient = np.reshape(self.output.gradient, self.input.getOutput().value.shape)
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class DropoutCell(Cell):
	def __init__(self, input, keepprob):
		self.input = input
		self.keepprob = keepprob
	def forwardPropagation(self):
		# 生成一个与输入同形的01分布矩阵，其中1的概率为keepprob
		self.keeparray = np.random.binomial(1, self.keepprob.getOutput().value, self.input.getOutput().value.shape)
		# 与分布矩阵相乘，从而“遮罩”某些数值
		self.output = var.Variable(self.input.getOutput().value * self.keeparray)
	def backwardPropagation(self):
		self.input.getOutput().gradient = self.output.gradient * self.keeparray
	def logOutputValue(self):
		print(self.output.value.shape, np.sum(self.output.value))
	def logOutputGradient(self):
		print(self.output.gradient.shape, np.sum(self.output.gradient))

class SoftmaxCell(SCell):
	def forwardPropagation(self):
		self.output = var.Variable(np.exp(self.input.getOutput().value) / np.sum(np.exp(self.input.getOutput().value)))
	def backwardPropagation(self):
		self.input.getOutput().gradient += -np.sum(self.output.gradient * self.output.value) * self.output.value + self.output.gradient * self.output.value;

class CrossEntropyCell(DCell):
	def forwardPropagation(self):
		self.output = var.Variable(-np.sum(np.log(self.input0.getOutput().value) * self.input1.getOutput().value))
	def backwardPropagation(self):
		self.input0.getOutput().gradient += self.output.gradient * -self.input1.getOutput().value / self.input0.getOutput().value
		self.input1.getOutput().gradient += self.output.gradient * -np.log(self.input0.getOutput().value)