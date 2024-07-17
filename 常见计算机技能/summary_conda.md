# Conda

### 终端上配置环境！
（1）两种打开方式<br/>
Anaconda Prompt中打开base环境，或在pycharm下方栏中使用特定环境内的终端

（2）常用命令-配置环境<br/>
查看base下所有环境：`conda env list`<br/>
创建新的虚拟环境：`conda create -n 虚拟环境名字 python=版本`<br/>
删除虚拟环境：`conda remove -n 虚拟环境名字 --all`<br/>
进入某环境 `conda activate 虚拟环境名`<br/>

（3）环境内包的配置<br/>
！！！`conda install`尽量不要用！！！用pip来安装更安全<br/>
查看环境内的包： `pip list / conda list`<br/>
配置torch： `nvidia-smi`查看cuda version版本，<br/>
    torch网上的cuda runtime版本（12.1）必须满足低于该cuda driver版本（12.3）<br/>
    `pip3 install torch torchvision torchaudio --index-url`<br/>

判断是否配好gpu：
```
python #先进入python环境
import torch
print(torch.cuda.is_available())
```
安装某个包：`pip install 包名==版本号`
<br/>卸载某个包：`pip uninstall 包名`


