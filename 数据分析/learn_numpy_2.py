import numpy
import numpy as np

print("4.向量化和广播")
#广播机制：在算术运算期间处理不同形状的数组，较小数组在较大数组上广播，以便具有兼容的性状
x = np.arange(4)
y = np.ones((3,4))
print(f'x:{x},\ny:{y}')
print(x+y)

#数学函数
print(numpy.add(x,1))
print(numpy.subtract(x,1))
print(np.multiply(x,2))
print(np.divide(x,2))
print(np.floor_divide(x,2))
print(np.power(x,2))

print(np.sqrt(x))
print(np.square)

print(np.sin(x))
#指数对数
print(np.exp(x))
print(np.log(y))
#加法乘法函数，通过axis设置方向
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.sum(x)
print(y)  # 575
y = np.sum(x, axis=0)
print(y)  # [105 110 115 120 125]
y = np.sum(x, axis=1)
print(y)  # [ 65  90 115 140 165]
y = np.prod(x, axis=0)
print(y) #[2978976 3877632 4972968 6294624 7875000]

#np.around舍入 np.ceil上限 np.floor下限 np.abs绝对值

print("5.排序搜索计数")

print("6.输入输出")

print("7.随机抽样")

print("8.统计相关")

print("9.线性代数")