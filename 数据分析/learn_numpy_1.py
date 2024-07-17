import numpy as np

print("1.1常量")
#numpy.nan 表示空值
#定义一个np数组
x = np.array([1,1,9,np.nan,10])
print(x)#[ 1.  1.  9. nan 10.]
y = np.isnan(x)
print(y)#[False False False  True False]
z = np.count_nonzero(y)
print(z)#1

print("1.2数据类型")
#int8,int16,int32,uint8(无符号整型)，float16...
a = np.dtype('b1')#boolen类型
print(a.type) #<class 'numpy.bool_'>
print(a.itemsize) #1
#！python整数类型范围无限，但numpy整数类型存储范围有限，超出会溢出，因此需指定类型
ii16 = np.iinfo(np.int16) #查看数据类型范围
print(ii16.min) #-32768
print(ii16.max) #32767
# #任意数组转为numpy格式时，可指定整数类型防止溢出
w = [400000]
c = np.array(w, dtype='int16')#超出边界有时会预警
print(c)
d = np.array(w, dtype='int64')
print(d)

print("1.3时间日期和时间增量")
print("1.4数组的创建")
d = np.array([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
print(d, type(d),d.shape)
#零数组、1数组和空数组
x = np.zeros(5) #[0. 0. 0. 0. 0.] 参数为大小shape
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.zeros_like(x)
print(y)# [[0 0 0],[0 0 0]]
x = np.ones(4) #[1,1,1,1]
print(x)
x = np.empty((1,2))
print(x)
#单位数组/矩阵,对角数组
x = np.eye(3)
print(x)
x = np.arange(9).reshape((3, 3))
print(x)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(np.diag(x))  # [0 4 8]
print(np.diag(x, k=1))  # [1 5]
print(np.diag(x, k=-1))  # [3 7]
v = [1, 3, 5, 7]
x = np.diag(v)
print(x)
# [[1 0 0 0]
#  [0 3 0 0]
#  [0 0 5 0]
#  [0 0 0 7]]
#常数数组
x = np.full(2, 7)
print(x) #[7,7]
#数值范围创建数组
x = np.arange(3, 7, 2) #从【到)固定间隔
print(x)  # [3 5]
x = np.linspace(start=0, stop=2, num=9)
print(x)  # [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]
x = np.random.random((2, 3))
print(x)
#结构数组：先定义结构再由np.array创建数组并确定其参数dtype
mytype = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})
mytype = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=mytype)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>

#数组的属性！！！itemsize以字节的形式返回数组中每一个元素的大小
a = np.array([(1,2,3,4,5),(6,7,8,9,10)])
print(a)
print(a.shape)  # (2,5)
print(a.dtype)  # int32
print(a.size)  # 10
print(a.ndim)  # 2
print(a.itemsize)  # 4

print("1.5切片与迭代")
#索引
print(a[1])
print(a[1,1])
#[起始:结束:间隔]
#二维数组第一片定义行 切片第二片定义列的切片
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
x[0::2,1::3] = 0
print(x) #！高维数组索引时只取[4,2,3]中的4进行索引，所以[2,3]部分不变

print("1.6数组操作与变形")
#1直接改变.shape属性
x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
x.shape = [2, 4]
print(x)
# [[1 2 9 4]
#  [5 6 7 8]]

#2flatten进行展平为1维
x = x.flatten()
print(x)
#3reshape赋予新形状
x = np.reshape(x,[1,4,2,1])
print(x)
x = np.reshape(x,[2,-1]) #-1为自动计算
print(x)
#4数组转置
print(x.T)
y = np.transpose(x)
print(y)
#5用np.newaxis加空维度，np.squeeze删空维度
x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
print(x)  # [1 2 9 4 5 6 7 8]
y = x[np.newaxis, :]
print(y.shape)  # (1, 8)
print(y)  # [[1 2 9 4 5 6 7 8]]
y = np.squeeze(x)
print(y.shape)  # (8,)

#6两数组组合
#concatenate为指定原维度拼接，stack为增加维度拼接
x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.concatenate([x, y],axis=0)
print(z)
# [1 2 3 7 8 9]
z = np.stack([x,y],axis=1)
print(z)#[[17][28][39]]

#7两数组拆分
#split，垂直拆分vsplit,水平切分hsplit
x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.split(x, [1, 3])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]

#8数组平铺
#tile重复，repeat
x = np.repeat(x,2)
print(x)
