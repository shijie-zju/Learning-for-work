# Network
1.数据集——>模型——>训练(正向、损失、反向、更新)——>推理<br/>
2.tensor格式定义参数：<br/>
w = torch.tensor([[1.0], [2.0], [3.0]]) #直接初始定义

#tensor即包含了w也包含w.grad，直接输出都是计算图，需要.data查看值，.item()输出非tensor格式
w.requires_grad = True  # 需要计算梯度

w.data = w.data - 0.01 * w.grad.data #更新梯度值
w.grad.data.zero_() #更新后梯度归零，否则下次再更新会按计算图叠加

3.以上都可以进行封装：
y_pred = model(x_data) #①使用模型正向传播 & 设置模型！
loss = criterion(y_pred, y_data) #②计算损失Loss & 损失函数！
print(epoch, loss.item())
optimizer.zero_grad() #梯度归零
loss.backward() #③计算梯度（Loss关于所有含grad的tensor都要计算）
optimizer.step() #④更新梯度 & 优化器！

4.①设置模型部分：
self.linear1 = torch.nn.Linear(8, 6) #其中（8,6）是输入输出，也是y（N*6）=激活（ x（N*8）*w(8*6)+b(1*6广播后N*6) ） 中w的维度
self.sigmoid = torch.nn.Sigmoid()
init中是初始化，目的是设定参数维度
forward中是前向传播网络，目的是接收特定维度的数据

5.②计算损失
criterion = torch.nn.MSELoss(reduction='sum') #均方误差，reduction='sum'是最后loss加和
criterion = torch.nn.BCELoss(size_average=False，reduction='mean') #二分类交叉熵损失，size_average=False对一个batch里面的所有的数据不求求均值
criterion = torch.nn.CrossEntropyLoss() #交叉熵损失，此时最后一层网络不需要激活了，
# 因为softmax会取到0-1且和为1的范围，相当于交叉熵包括了 softmax(x)+log(x)+nn.NLLLoss负对数似然损失===>nn.CrossEntropyLoss
6.数据集读取
epoch（几轮），数据集大小=batchsize（一次更新多少）*iteration（指一轮更新几次）
dataloader 主要用于从数据集中拿出batchsize大小的数据 # y（batch*6）= 激活（x(batch*8) * w(8*6) + b）

(1)自定义数据集读取数据：
#dataset
class DiabetesDataset(Dataset): #继承并创建新的dataset类
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32) #以,连接，加载数据
        self.len = xy.shape[0]  # shape(多少行，多少列)

        #数组numpy转张量！！！
        self.x_data = torch.from_numpy(xy[:, :-1]) #取出x并将生成的 数组numpy转tensor！！！
        self.y_data = torch.from_numpy(xy[:, [-1]]) #取出y并转numpy

        #dataframe转numpy！！！
        #DataFrame.values方法可以将DataFrame转换为NumPy数组，我们可以进一步将它转换为Tensor。
        xy = df_xy.values #dataframe转numpy！！！


    def __getitem__(self, index):  #!!!可以通过定义getitem，后面调用时直接取出x和y
        return self.x_data[index], self.y_data[index]
dataset = DiabetesDataset('diabetes.csv')

#dataloader
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)  #shuffle多线程，num_workers 多线程
（2）调用现成的库中的数据集
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

（3）进行封装后基于loader中的数据训练
#使用多线程最好加if行将下面封装，不然报错
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  #train_loader取出的data是一个batch的tensor形式，可以拆成输入和标签,0是指索引i从0开始，可改为1
            inputs, labels = data

（4）transform
#处理图像数据输入，一般在getitem时候调用transform
#在dataset读数据前，需要将取值范围从[0,255]大小28*28转换为tensor格式的[0,1]大小1通道*28*28！！！ 然后归一化(设置均值，标准差)变成01分布
transform = transforms.Compose([ transforms.ToTensor() , transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

（5）维度转变
transforms.ToTensor(), #（N*28*28）数组 ——> (N*1*28*28)张量
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)  # (N*1*28*28)张量 ——> iteration个（batchsize*1*28*28）
x = x.view(-1, 784)  # batchsize*1*28*28 ——> batchsize*1*784

7.交叉熵
（1）softmax层保障了多分类问题中， 输出大于0 ， 且和等于1
（2）softmax后再进行log，然后与onehot标签相乘，最后相当于-Ylog(Yhat)

8.CNN网络
CNN —— GRU —— LSTM
(1)正常网络搭建
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
        batch_size = x.size(0)  # 取x的size中最外维度为batchsize，输入batchsize*1*28*28
        x = F.relu(self.pooling(self.conv1(x)))  # 卷积核是5，所有W和H都-4 ——> batchsize*10*24*24 ——> batchsize*10*12*12
        x = F.relu(self.pooling(self.conv2(x)))  # ——> batchsize*20*8*8 ——> batchsize*20*4*4
        x = x.view(batch_size, -1)  # 转为batch*展平的层，用于全连接分类 ——> batchsize*320
        # print("x.shape",x.shape)
        x = self.fc(x)  # ——> batchsize*10
        return x
(2)1*1卷积
作用：仅可以改变通道数，W*H不变，降低运算量。
因为192*28*28 ——> 32*28*28 正常5*5卷积计算量：5*5*28*28*192(原所有通道，叠加)*32(新的通道要准备的核数量)
先一维卷积192*28*28 ——>(1*1卷积) 16*28*28——>(5*5卷积)32*28*28 计算量：1*1*28*28*192*16+5*5*28*28*16*32降低了10倍
(3)使用class inceptionA(nn.Module) 构造网络块，可以更好地减少复杂网络编写时的代码冗余
(4)class ResidualBlock(nn.Module)
问题：网络层数加多，效果反而下降，因为反向传播过程中需要链式传播去把一个个梯度乘起来，越乘越小梯度容易趋近0，出现 梯度消失
解决：Residual块中设定 H(x)=F(x)+x（可见块输入x和输出时F(x)的张量维度必须一样）,反向传播时求梯度为 F对x偏导(接近0)+1 ,接近1所以不会越乘越小
补充：不同ResidualBlock设计可以不同，只要至少保证输入输出同维度

9.调用GPU
#定义模型对象时
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#从dataloader中取出样本和标签时
for batch_idx, data in enumerate(train_loader, 0):
    inputs, target = data
    inputs, target = inputs.to(device), target.to(device)

10.RNN网络：处理具有序列连接关系的数据，如股市、天气、自然语言等序列

卷积神经网络是3维输入      (                            batch张图                         *通道C*W*H)中的不同batch*W*H区域共享一个batch通道的卷积核的参数
循环神经网络是时序+新3维输入(seqlen(序列长度，即一段话的字数)*batch段话(batch=1一次更新只输入一段话)*特征C)    中的不同batch*C共享一个全连接层的网络参数

(1)如何word2vec
①独热向量：一个字母可以表示成[0,0,0,0,1,0,0,0],维度=字母种类数input_layers
    缺点：种类过多的话，维度就太高了，会面对维度诅咒，且矩阵稀疏，且是硬编码的不是学习出来的，希望变成低维稠密可学习的
②嵌入层embedding：独热向量乘以 (原inputsize(字符类数),embeddingsize) 的矩阵(降维),最后一个字符对应的是一个 (1,embeddingsize) 的向量
    要求：embedding层输入是longtensor的张量，输入（seqlen,batchsize）增加一个维度！输出（seqlen,batchsize，hiddensize）

(2)维度要求
①输入size
inputs = torch.LongTensor(x_data)  #x_data (seqlen一段长, batchsize共几段, inputsize降维后单字符的特征维度)
或者self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True) #batch_first=True,输入要求改为(batchsize,seqlen,inputsize)
②embedding层会升维！
定义torch.nn.Embedding(input_size=3(即0-2需要映射到空间), embedding_size=10)
输入a:(1,3)tensor([[1, 0, 2]])
输出b:(1,3,10)tensor([[[-1.3087, -0.3111, -1.2043,  0.0683, -0.5638, -1.5501,  1.1962,
          -0.5944,  1.2391,  0.3474],
         [ 1.4109,  0.4483,  0.9972, -2.1846,  0.3635, -1.9141,  2.0747,
          -1.3027,  0.6383,  1.2249],
         [-1.2505,  1.2822, -0.7035,  2.0647, -0.3783,  0.0063,  1.0370,
          -1.0674, -0.8313,  0.1022]]

11.进阶RNN实践
(1)定义dataset
获取数据集(参数确定是训练集还是测试集) —— 提取特征和标签列 —— 构造标签列表，获取标签数量 —— 构造标签字典，获得标签对应的数值
访问时，特征行 —— 对应行的标签值 —— 由字典获得数值
#还有定义maketensors函数
for i, (names, countries) in enumerate(trainLoader, 1):
    inputs, seq_lengths, target = make_tensors(names, countries)
    output = classifier(inputs, seq_lengths)

(2)定义模型的forward中
embedding = self.embedding(input) # result of embedding with shape：(seqLen, batchSize, hiddenSize)
# pack them up. 为了提高运行效率
gru_input = pack_padded_sequence(embedding, seq_lengths) #hiddensize长度不同，在一个batch中： 按长度排列 —— 记录结束位置 —— 一次计算时获取相应的参数长度


12.可视化
import matplotlib.pyplot as plt
acc_list = []
for epoch in range(1, N_EPOCHS + 1):
# Train cycle
    trainModel()  # 模型训练
    acc = testModel()  # 模型测试
    acc_list.append(acc)  # 存储测试准确率

epoch = np.arange(1, len(acc_list) + 1, 1)
plt.plot(epoch , acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show(block=True) #必加block=True！

13.训练计时
#定义
import math
import time
def time_since(sice):
    s = time.time() - sice      # 当前时间减去开始时间，s为单位
    m = math.floor(s / 60)      # 转成分钟
    s -= m*60
    return '%dm %ds' % (m, s)
# 在所有train开始前，记录训练开始时间
start = time.time()
#在train的过程中，记录从开始到该时刻时间
def trainModel():
    for i, (names, countries) in enumerate(trainLoader, 1):
        total_loss += loss.item()
        if i % 10 == 0:
            print(f' [{time_since(start)}] Epoch {epoch} ', end='')
