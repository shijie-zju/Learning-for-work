# matplotlib

```
import matplotlib.pyplot as plt
import numpy as np
```
### matplotlib使用时报错问题的解决
针对python=3.9版本： pip下载matplotlib==3.5.0<br/>
调用该包进行绘图前，运行下面的代码<br/>
```
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #不设置会卡死
#x = np.arange(1, 10)
#y = np.arange(1, 10)
#plt.plot(x,y)
plt.show(block=True) #block=True加上，否则显示图后会卡死
```

## 简单绘图
Matplotlib图像画在figure上，每个figure包含一个或多个axes子区域

1.创建普通画图 

`plt.plot([1,2,3,4],[1,4,2,3])`

2.创建包含axes的figure 

```commandline
fig, ax = plt.subplots() 
ax.plot([1,2,3,4],[1,4,2,3],label='red')
```

### 示例
```commandline
x = np.linspace(0, 2, 100)

fig, ax = plt.subplots()  
ax.plot(x, x, label='linear')  
ax.plot(x, x**2, label='quadratic')  
ax.plot(x, x**3, label='cubic')  
ax.set_xlabel('x label') 
ax.set_ylabel('y label') 
ax.set_title("Simple Plot")  
ax.legend() #图例
```
```commandline
x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear') 
plt.plot(x, x**2, label='quadratic')  
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
```

## 详细解读
和人作画的步骤类似：

准备一块画布或画纸

准备好颜料、画笔等制图工具

作画

所以matplotlib有三个层次的API：

matplotlib.backend_bases.FigureCanvas 代表了绘图区，所有的图像都是在绘图区完成的

matplotlib.backend_bases.Renderer 代表了渲染器，可以近似理解为画笔，控制如何在 FigureCanvas 上画图。

**matplotlib.artist.Artist** 代表了具体的图表组件，即调用了Renderer的接口在Canvas上作图。
前两者处理程序和计算机的底层交互的事项，第三项Artist就是具体的调用接口来做出我们想要的图，比如图形、文本、线条的设定。所以通常来说，我们95%的时间，都是用来和matplotlib.artist.Artist类打交道的。

---
Artist有两种类型：primitives 和containers。

primitive是基本要素，它包含一些我们要在绘图区作图用到的标准图形对象，如曲线Line2D，文字text，矩形Rectangle，图像image等。

container是容器，即用来装基本要素的地方，包括图形figure、坐标系Axes和坐标轴Axis。

---
matplotlib的标准使用流程为：

1. 创建一个Figure实例
2. 使用Figure实例创建一个或者多个Axes或Subplot实例（坐标轴/子图）
3. 使用Axes实例的辅助方法来创建primitive

完整过程如下
```commandline
import matplotlib.pyplot as plt
import numpy as np

# step 1 
# 我们用 matplotlib.pyplot.figure() 创建了一个Figure实例
fig = plt.figure()

# step 2
# 然后用Figure实例创建了一个两行一列(即可以有两个subplot)的绘图区，并同时在第一个位置创建了一个subplot
ax = fig.add_subplot(2, 1, 1) # two rows, one column, first plot

# step 3
# 然后用Axes实例的方法画了一条曲线
t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line, = ax.plot(t, s, color='blue', lw=2)
```

1.设置对象的属性Artist

属性的实例都存储在成员变量(member variables) Figure.patch 和 Axes.patch中。 
（Patch是一个来源于MATLAB的名词，它是图形上颜色的一个2D补丁，包含rectangels-矩形，circles-圆 和 plygons-多边形）

查看 
```commandline
plt.figure().patch
plt.axes().patch
```
修改
```commandline
a = o.get_alpha()
o.set_alpha(0.5*a)
```

2.如何获取对象

primitive是基本要素，它包含一些我们要在绘图区作图用到的标准图形对象，如曲线Line2D，文字text，矩形Rectangle，图像image等。

1. 2DLine

常用的的参数有：

xdata:需要绘制的line中点的在x轴上的取值，若忽略，则默认为range(1,len(ydata)+1)

ydata:需要绘制的line中点的在y轴上的取值

linewidth:线条的宽度

linestyle:线型

color:线条的颜色

marker:点的标记，详细可参考markers API

markersize:标记的size

**使用方法**

a.直接设置
```commandline
import matplotlib.pyplot as plt
x = range(0,5)
y = [2,5,7,8,10]
plt.plot(x,y, linewidth=10) # 设置线的粗细参数为10
```
b.获得线属性，使用setp函数设置
```commandline
x = range(0,5)
y = [2,5,7,8,10]
lines = plt.plot(x, y)
plt.setp(lines, color='r', linewidth=10)
```


2. Patches(二维图形类)

A. Rectangle矩形

a.hist 直方图
```commandline
import matplotlib.pyplot as plt
import numpy as np 
x=np.random.randint(0,100,100) #生成【0-100】之间的100个数据,即 数据集 
bins=np.arange(0,101,10) #设置连续的边界值，即直方图的分布区间[0,10],[10,20]... 
plt.hist(x,bins,color='fuchsia',alpha=0.5)#alpha设置透明度，0为完全透明 
plt.xlabel('scores') 
plt.ylabel('count') 
plt.xlim(0,100)#设置x轴分布范围 plt.show()
```
b.bar 柱状图
```commandline
import matplotlib as mpl
y = range(1,17)
plt.bar(np.arange(16), y, alpha=0.5, width=0.5, color='yellow', edgecolor='red', label='The First Bar', lw=3)
```
B.Polygon 多边形
```commandline
import matplotlib.pyplot as plt
x = np.linspace(0, 5 * np.pi, 1000) 
y1 = np.sin(x)
y2 = np.sin(2 * x) 
plt.fill(x, y1, color = "g", alpha = 0.3)
```

C.Wedge 契形
```commandline
import matplotlib.pyplot as plt 
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10] 
explode = (0, 0.1, 0, 0) 
fig1, ax1 = plt.subplots() 
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90) 
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle. 
plt.show()
```

3. Collection

collections类是用来绘制一组对象的集合，collections有许多不同的子类，如RegularPolyCollection, CircleCollection, Pathcollection, 分别对应不同的集合子类型。其中比较常用的就是散点图，它是属于PathCollection子类，scatter方法提供了该类的封装，根据x与y绘制不同大小或颜色标记的散点图。

```commandline
x = [0,2,4,6,8,10] 
y = [10]*len(x) 
s = [20*2**n for n in range(len(x))] 
plt.scatter(x,y,s=s) 
plt.show()
```
4. images

images是matplotlib中绘制image图像的类，其中最常用的imshow可以根据数组绘制成图像
```commandline
import matplotlib.pyplot as plt
import numpy as np
methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


grid = np.random.rand(4, 4)

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(grid, interpolation=interp_method, cmap='viridis')
    ax.set_title(str(interp_method))

plt.tight_layout()
plt.show()
```
4.对象容器 - Object container
```commandline
fig = plt.figure()
ax1 = fig.add_subplot(211) # 作一幅2*1的图，选择第1个子图
ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.3]) # 位置参数，四个数分别代表了(left,bottom,width,height)
print(ax1) 
print(fig.axes) # fig.axes 中包含了subplot和axes两个实例, 刚刚添加的
```

## 案例
1. 多线图

```commandline
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 导入数据集并转成方便作图的格式
Dataset = pd.read_csv('data/Drugs.csv')
group = Dataset.groupby(['YYYY','State']).agg('sum').reset_index()
df = group.pivot(index='YYYY', columns='State', values='DrugReports').reset_index()

# 设定式样
plt.style.use('seaborn-darkgrid')
 
# 创建调色板， 色卡用来控制每条线的颜色
palette = plt.get_cmap('Set1')

# 绘图
plt.figure(figsize=(15, 7))
num=0
for column in df.drop('YYYY', axis=1):
    num += 1
    plt.plot(df['YYYY'], df[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)
    
plt.legend(loc=2, ncol=2)
plt.title("Multiple line plot", loc='center', fontsize=12, fontweight=0, color='orange')
plt.xlabel("year")
plt.ylabel("DrugReports")
plt.show()
```

2. 多子图

```commandline
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 导入数据集并转成方便作图的格式
Dataset = pd.read_csv('data/Drugs.csv')
group = Dataset.groupby(['YYYY','State']).agg('sum').reset_index()
df = group.pivot(index='YYYY', columns='State', values='DrugReports').reset_index()

# 初始化画布的设定
plt.style.use('seaborn-darkgrid') # 风格
palette = plt.get_cmap('Set1') # 颜色卡
plt.figure(figsize=(15, 10)) # 画布大小

# 绘制
num=0
for column in df.drop('YYYY', axis=1):
    num+=1
 
    # 设定子图在画布的位置
    plt.subplot(3,3, num)
 
    # 画线图
    plt.plot(df['YYYY'], df[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
 
    # 设定子图的X轴和Y轴的范围，注意，这里所有的子图都是用同一套X轴和Y轴
    plt.xlim(2009.3,2017.3)
    plt.ylim(0,50000)
 
    # 添加每个子图的标题
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

# 添加整个画布的标题
plt.suptitle("How many DrugReports the 5 states have in past few years?", fontsize=13, fontweight=0, color='black', style='italic', y=0.95)
 
# 添加整个画布的横纵坐标的名称
plt.text(2014, -9500, 'Year', ha='center', va='center')
plt.text(1998, 60000, 'DrugReports', ha='center', va='center', rotation='vertical')

```

更多案例：https://tianchi.aliyun.com/course/324/3659