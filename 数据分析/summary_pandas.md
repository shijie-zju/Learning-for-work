# pandas

```commandline
#导入os库，执行各种与文件系统和操作系统相关的任务
import os
#导入pandas库，里面的dataframe仿照了R语言的系列功能
import pandas as pd
```
列表
```commandline
lista = [] #创建一个空列表
lista.append(row_lista) #将列表数据按行添加

lista[1] #访问列表第2行/项
```
数据框dataframe
```commandline
frame1 = pd.DataFrame() #创建一个新的数据框
frame2 = pd.DataFrame(lista) #将列表转换为数据框
frame3 = pd.DataFrame(0, index=range(frame1.shape[0]), columns=frame1.columns) #创建新的frame,元素0,行名和列名定义
frame4 = pd.concat([ frame1, frame2[某列] ], axis=1) #按列axis=1进行拼接

frame1.shape[0] #行的个数
frame1.shape[1] #列的个数
frame1.index #所有行索引名
frame1.columns #所有的列名

for column in frame1.columns: #遍历所有的列
for column_name, column_data in frame1.iteritems():  # column_name 是列名，column_data 是包含列数据的 Series
for i in range(0,frame1.shape[0]): #从第0列挨个访问
for index, row in frame1.iterrows():  #遍历所有行：index 是行索引，row 是包含行数据的 Series

frame1["列名"] #取某列，column1为列名
frame1[col] #2则是取第3列
frame1.loc["行标签"] #取某行，loc看名，iloc看数字
frame1.iloc[row] #2则是取第3行 

df.iloc[row,col] #取某个值，

frame1.mean() #每列平均值
frame1.mean().mean() #再平均则为整个frame均值
new_column = frame1.apply(mean, axis=1) #apply将函数通过apply应用到每一行
value = frame1.apply(lambda x: x**p).sum() #每列中每个值先操作p次方，再统一求和
```
文件读取
```commandline
def dan_read(folder_path,dan_df):
    # 获取文件夹中所有txt文件的文件名
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')] #os.listdir：列出路径下的子目录
    # 遍历每个txt文件并将其读入数据框，然后将数据框存储在dataframes列表中
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file) #os.path.join:自动创建适当的路径分隔符,并合并
        # 使用pandas的read_csv方法读取txt文件
        data=[]
        with open(file_path, 'r') as file: #打开文件只读
            for line in file:# 从每行中分割数据（以空格分隔）并添加到数据列表
                row_data = line.strip().split(' ') #以split(' ')空格分割，strip：字符串方法，用于删除字符串两端的空白字符
                data.append(row_data) #放入data列表中
            # 将数据列表转换为DataFrame
            df = pd.DataFrame(data)
            #转换后数据类型为object，需要转换为int64
            for column in df.columns:
                if df[column].dtype == 'object':
                    try:
                        df[column] = pd.to_numeric(df[column], downcast='integer')
                    except ValueError:# 处理无法转换为整数的情况
                        pass
                    
            dan_df.append(df)
    return dan_df
```