from mnist import MNIST
import numpy as np
import random
import variable as var
import cell
import network

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

# 构造网络

cnn = network.Network();
#cnn.logForwardPropgation = True
#cnn.logBackwardPropagation = True

# 输入层
X = var.Variable(np.zeros([28, 28, 1]))
Y = var.Variable(np.zeros([1, 10]))
KeepProb = var.Variable(0)

# 卷积层 I
WC1 = var.Variable(np.random.randn(5, 5, 1, 32) / 10)
cnn.appendVariable(WC1)

convCell1 = cell.ConvolutionCell(X, WC1) # => [28, 28, 32]
cnn.appendCell(convCell1)

# 激活层 I
BC1 = var.Variable(np.ones(32) / 10)
cnn.appendVariable(BC1)

reluCell1 = cell.ReLuCellType1(convCell1, BC1) # => [28, 28, 32]
cnn.appendCell(reluCell1)

# 池化层 I
mpCell1 = cell.MaxPoolingCell(reluCell1, [2, 2]) # => [14, 14, 32]
cnn.appendCell(mpCell1)

# 卷积层 II
WC2 = var.Variable(np.random.randn(5, 5, 32, 64) / 10)
cnn.appendVariable(WC2)

convCell2 = cell.ConvolutionCell(mpCell1, WC2) # => [14, 14, 64]
cnn.appendCell(convCell2)

# ReLu Layer 2
BC2 = var.Variable(np.ones(64) / 10)
cnn.appendVariable(BC2)

reluCell2 = cell.ReLuCellType1(convCell2, BC2) # => [14, 14, 64]
cnn.appendCell(reluCell2)

# Max-Pooling Layer 2
mpCell2 = cell.MaxPoolingCell(reluCell2, [2, 2]) # => [7, 7, 64]
cnn.appendCell(mpCell2)

# Re-Shape Layer 
rsCell = cell.ReShapeCell(mpCell2, [1, 7 * 7 * 64]) # => [1, 7 * 7 * 64]
cnn.appendCell(rsCell)

# Full-Connected Layer 1
WFC1 = var.Variable(np.random.randn(7 * 7 * 64, 1024) / 10)
cnn.appendVariable(WFC1)

matmulCell1 = cell.MatMulCell(rsCell, WFC1) # => [1, 1024]
cnn.appendCell(matmulCell1)

BFC1 = var.Variable(np.ones(1024) / 10)
cnn.appendVariable(BFC1)

reluCell3 = cell.ReLuCellType2(matmulCell1, BFC1)
cnn.appendCell(reluCell3)

# Drop-out Layer
KeepProb = var.Variable(np.array(0)) # Input

dropoutCell = cell.DropoutCell(reluCell3, KeepProb) # => [1, 1024]
cnn.appendCell(dropoutCell)

# Full-Connected Layer 2
WFC2 = var.Variable(np.random.randn(1024, 10) / 10)
cnn.appendVariable(WFC2)

matmulCell2 = cell.MatMulCell(dropoutCell, WFC2) # => [1, 10]
cnn.appendCell(matmulCell2)

BFC2 = var.Variable(np.ones([1, 10]) / 10)
cnn.appendVariable(BFC2)

addCell = cell.AddCell(matmulCell2, BFC2) # => [1, 10]
cnn.appendCell(addCell)

# Softmax
softmaxCell = cell.SoftmaxCell(addCell) # Softmax(X * W + B) => [10, 1]
cnn.appendCell(softmaxCell)

# Loss (Cross-entropy)
loss = cell.CrossEntropyCell(softmaxCell, Y) # CrossEntropy(Softmax(X * W + B), Y) => Loss
cnn.appendCell(loss)

# Training
BATCH_NUMBER = 20000 # BATCH的数量
BATCH_SIZE = 50 # BATCH的大小
LEARNING_RATE = 0.0001 #学习速率
for batch_index in range(BATCH_NUMBER):
    # 构造一个BATCH
    batch_xs = [];
    batch_ys = [];
    for data_index in range(BATCH_SIZE):
        # j = random.randint(0, len(images) - 1)
        j = data_index
        x = images[j];
        y = labels[j];
        batch_xs.append(x);
        batch_ys.append(y);

    # 使用这个BATCH进行训练
    batch_loss = 0
    batch_precision = 0
    for data_index in range(BATCH_SIZE):
        x = batch_xs[data_index];
        y = batch_ys[data_index];
        X.takeInput(x);
        Y.takeInput(y);
        KeepProb.takeInput(np.array(0.5));
        cnn.forwardPropagation() # 正向传播
        batch_loss += loss.getOutput().value # 统计整个BATCH的损失
        loss.getOutput().gradient = -1 / BATCH_SIZE # 整个BATCH统一计算梯度，所以单个数据点的输出梯度只有1/BATCH_SIZE
        cnn.backwardPropagation() # 反向传播
        # print('    data', data_index, ', loss =', loss.getOutput().value)
        predict = np.argmax(softmaxCell.getOutput().value)
        if predict == np.argmax(y):
        	batch_precision += 1 / BATCH_SIZE

    # 应用梯度
    cnn.applyGradient(LEARNING_RATE)
    print('batch', batch_index, ', loss =', batch_loss, ', precision =', batch_precision)

# Test
precision = 0
for index in range(len(images_test)):
    x = images_test[index];
    y = labels_test[index];
    X.takeInput(x);
    Y.takeInput(y);
    KeepProb.takeInput(np.array(1));
    cnn.forwardPropagation()
    predict = np.argmax(softmaxCell.getOutput().value)
    if predict == np.argmax(y):
        precision += 1 / len(images_test)
print('Precision =', precision)