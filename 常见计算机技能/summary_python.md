# Python

### 集合set：无序且不含重复元素
```commandline
my_set = {1,2,3} #两种方法创建集合 1大括号 
my_set = set([1,2,3]) #2将列表中元素转换为集合
```
添加 `my_set.add(x)`

创建 `a = set()`

成员测试 `x in my_set`

删除 `my_set.remove(x)` **`my_set.discard(x)`这个不存在时不报错**

长度 `len(my_set)`

### 列表list：有序有索引
`my_list = [1,2,3]`

添加元素 `my_list.append(x)` `my_list.insert(i,x) #i位置插入x`

索引 `my_list[0]`

删除`del my_list[2]`

创建空列表 `mylist = []`

##### 字符列表

字符串转换为字符列表格式 `str_list = list('hello')`

字符列表排序 sort_list = `sorted(str_list)`

字符列表拼为字符串 `k = ''.join(sorted(list(str))))`

遍历
```commandline
for item in my_list:
    print(item)
```

切片 `my_list[start:end:step]`

长度 `len(my_list)`

### 字典dict：键值对其中键唯一
```commandline
my_dict = {'name': 'Alice', 'age': 25} #两种方式创建
my_dict = dict(name='Alice', age=25)
```
添加 `my_dict['gender']='male'`

访问 `value = my_dict['age']`

创建空字典 mydict = {}

遍历
```commandline
for key in my_dict:
    print(key)
for value in my_dict.values():
    print(value)
for key, value in my_dict.items():
    print(key, value)
```
**删除 `del my_dict['age']`** `my_dict.pop('age',[default])`

长度 `len(my_dict)`

### 元组tuple：创建后即不可变不可增删改
```commandline
my_tuple = (1, 2, 3) #不止1个元素则不用加逗号结尾
single_element_tuple = (1,)  # 注意逗号
```
访问 `element = my_tuple[1]`

切片  `my_tuple[start:end:step]`

长度 `len(my_tuple)`

### 队列deque：先进先出
```commandline
from collections import deque
my_queue = deque()
```
添加 `my_queue.append(x) #队尾入队` `my_queue.extend(iter): 将一个可迭代对象的所有元素添加到队列的末尾。`

删除 `my_queue.popleft() #移除并返回队列头第一个元素`

查看队列前端元素 `my_queue[0] #看即将出队被移除的元素`

查看队列后端元素 `my_queue[-1] #看刚入队的最后一个元素`

长度 `len(my_queue)`

### 正则表达式

Python中的正则表达式（Regular Expression）是一种强大的文本处理工具，可以匹配、搜索、替换或拆分复杂的字符串模式。以下是一些常见的Python正则表达式：

1. 匹配特定字符：

.：匹配任何字符（除了换行符）。<br/>
^：匹配输入字符串的开始。<br/>
$：匹配输入字符串的结束。<br/>
\d：匹配任何数字，等同于 [0-9]。<br/>
\D：匹配任何非数字字符，等同于 [^0-9]。<br/>
\s：匹配任何空白字符（包括空格、制表符、换页符等）。<br/>
\S：匹配任何非空白字符。<br/>
\w：匹配任何字母或数字或下划线，等同于 [a-zA-Z0-9_]。<br/>
\W：匹配任何非字母、非数字和非下划线的字符，等同于 [^a-zA-Z0-9_]。<br/>

2. 重复字符：

*：匹配前面的子表达式零次或多次。<br/>
+：匹配前面的子表达式一次或多次。<br/>
?：匹配前面的子表达式零次或一次。<br/>
{n}：n是一个非负整数。匹配确定的 n 次。<br/>
{n,}：n 是一个非负整数。至少匹配 n 次。<br/>
{n,m}：m 和 n 均为非负整数。最少匹配 n 次且最多匹配 m 次。<br/>

3. 选择、分组和引用：

|：表示或者，比如 a|b 匹配 'a' 或 'b'。<br/>
( )：将几个项组合为一个单元，例如 (abc) 与 abc 匹配相同的内容。捕获的内容可以由 \1,\2,\3... 等进行引用。<br/>
\：转义特殊字符，例如 \() 表示匹配真实的“(”字符，而不是作为分组符。<br/>

4. 预定义模式：

\d+ 或 \D+：匹配一个或多个数字或非数字字符。<br/>
\s+ 或 \S+：匹配一个或多个空白或非空白字符。<br/>
.：在 re 模块中，. 不能直接使用，因为它被视为一个特殊字符。如果要匹配任意字符（包括换行符），可以使用诸如 [\s\S] 或 [^\s] 的模式。<br/>

5. 边界条件：

^：在方括号外面表示否定，也可以表示字符串的开始。在方括号内表示非负整数，例如，[0-9]^ 表示以0开头的一串数字。<br/>
$：表示字符串的结束，也可以表示美元符号。在方括号内表示负整数，例如，[-1]^ 表示以-1结尾的一串数字。<br/>

6. 贪婪与非贪婪匹配：

默认情况下，正则表达式是贪婪的，即它们尽可能多地匹配（只要还能符合其他要求）。可以使用 ? 来使正则表达式变为非贪婪的（尽可能少地匹配）。例如，在查找所有以 "a" 开头的单词时，"a*" 将匹配尽可能多的 "a" 字符，"a*?" 则将只匹配最少的 "a" 字符以满足条件