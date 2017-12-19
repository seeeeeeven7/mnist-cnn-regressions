import variable

class Network:
    def __init__(self):
    	# 该网络中的参数和计算单元
        self.variables = []
        self.cells = []
        # 是否对参数和梯度传递过程输出log
        self.logForwardPropgation = False
        self.logBackwardPropagation = False
    def getVariablesAmount(self):
        return len(self.variables);
    def getCellsAmount(self):
        return len(self.cells);
    # 添加新的参数
    def appendVariable(self, variable):
        self.variables.append(variable);
    # 添加新的计算单元
    def appendCell(self, cell):
        self.cells.append(cell);
    # 前向传播过程
    def forwardPropagation(self):
        for cell in self.cells:
            if self.logForwardPropgation:
                print(cell.__class__.__name__)
            cell.forwardPropagation()
            if self.logForwardPropgation:
                cell.logOutputValue()
    # 反向传播过程
    def backwardPropagation(self):
        for cell in reversed(self.cells):
            if self.logBackwardPropagation:
                print(cell.__class__.__name__)
                cell.logOutputGradient()
            cell.backwardPropagation();
    # 应用梯度过程
    def applyGradient(self, step_size):
        for variable in self.variables:
            variable.applyGradient(step_size);