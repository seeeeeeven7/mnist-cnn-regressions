import numpy as np

class Variable:
    def __init__(self, value = None): # 根据value构造一个同形的梯度gradient
        if value is not None:
            self.value = value
            self.gradient = np.zeros_like(value, dtype=np.float) # 注意注明值的类型为浮点数
        else:
            self.gradient = None
    def __str__(self):
        return '{ value =\n' + str(self.value) + ',\n gradient =\n' + str(self.gradient) + '}';
    def __repr__(self):
        return self.__str__();
    def getOutput(self):
        return self
    def takeInput(self, value): # 仅在该变量为输入时调用
        self.value = np.reshape(value, self.value.shape);
        self.gradient = np.zeros_like(self.value, dtype=np.float)
    def applyGradient(self, step_size): # 仅在该变量为参数时调用
        self.value = self.value + self.gradient * step_size
        self.gradient = np.zeros_like(self.gradient, dtype=np.float)