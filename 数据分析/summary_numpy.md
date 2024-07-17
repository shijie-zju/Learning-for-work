# numpy
`import numpy as np`

### 创建数组
`np.array([[(1,2),(3,4)],[(5,6),(7,8)]],dtype='int16')`

创建其他类型数组: 

零数组 `np.zeros((3,2)) #shape是(3,2)`

一数组 `np.ones(4)`

空数组 `np.empty`

单位数组 `np.eye`

对角数组 `np.diag(x,k=1) #前提x是n*n的矩阵,k=1为往上一排对角`

常数数组 `np.full(shape=2,fill_value=5)`

数值范围创建数组 `np.arange(start=3,stop=7,step=2) #步长`

数值范围创建数组 `np.linspace(start=3,stop=7,num=4) #数量`

数值随机数创建 `np.random.randint((2,3))`

### 查看类型
`a.shape #(2,5)`

`a.dtype #int32`

`a.size #大小10`

`a.ndim #维度2`

`a.itemsize #字节形式每一元素大小`

索引 `a[0::2,1::3] #[起：终：隔],第1片定义行,第2片定义列`

### 变形
直接重新设置结构属性 `x.shape = [2,3]`

展平为1维 `x.flatten()`

赋予新形状 `np.reshape(x,newshape:[1,2,3,-1])`

数组转置 `x.T` 或 `np.transpose(x)`

加空维度 `x[np.newaxis,:]`

删空维度 `np.squeeze(x)`

两数组拼接(指定原维度) `np.concatenate([x,y],axis=0) #axis为拼接的维度`

两数组拼接(增加维度拼) `np.stack([x,y],axis=1)`

### 广播与计算
加 `np.add(x,1)`

减 `np.subtract(x,1)`

乘 `np.multipy(x,1)`

除 `np.divide(x,1)`

乘方 `np.power(x,1)`

开根 `np.sqrt(x)`

三角 `np.sin(x)`

指数 `np.exp(x)`

对数 `np.log(x)`

矩阵自身计算行/列和 `np.sum(x,axis=0)`

矩阵自身计算行/列乘 `np.prod(x,axis=0)`

两向量间相乘并相加 `np.dot(a,b)`
### 排序搜索计数
axis=0，1 指行和列，但排序时每行间做对比，实则一列一列去排序

排序后显示结果 `np.sort(a,axis=1,kind='quicksort',order=None) #指列上元素，按行排！类型为快速排序法，order是默认字段名`

排序后显示索引 `np.argsort(a)`

排序按照某指标 `np.lexsort(a[:,0])`

查找 `np.where(x<5,x,10*x) #(条件，满足则操作，不满足则操作)`

计数 `np.count_nonzero(a)`

### 统计相关
最小值 `np.amin(x)`

最大值 `np.amax(x)`

极差 `np.ptp(x,axis=0)`

分位数 `np.percentile(x,[25,50]) #计算25，75处的分位数`

中位数 `np.median(x,axis=0)`

平均值 `np.mean(x,axis=0)`

加权平均值 `np.average(x,axis=0)`

方差 `np.var(x)`

标准差 `np.std(x)`

协方差 `np.cov(x)`

相关系数 `np.corrcoef(x,y)`

### 线性代数
矩阵乘积或向量内积 `np.dot(a,b)`

特征值特征向量 `a,b = np.linalg.eig(x)`

矩阵范数 `np.linalg.norm(x,ord=1) #ord决定了计算什么范数`

方阵行列式 `np.linalg.det(x)`

矩阵的秩 `np.linalg.matrix_rank(x)`

矩阵的逆 `np.linalg.inv(A)`



