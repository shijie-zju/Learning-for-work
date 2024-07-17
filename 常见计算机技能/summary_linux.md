## 常用linux命令
`cd` 进入一个目录（..上一个目录；-上次位置）

`ls` 列出当前目录所有文件（-l所有属性；-a显示隐藏文件）

`ll` 列出所有信息

`pwd` 打印当前路径

`mkdir` 创建文件夹（后直接 +文件夹名称）

## vim
`vim` 输入后进入编辑器可编辑文件（ctrl x退出 ctrl o保存）

`:e filename` 打开文件

`u ctrl+4` 撤销

`:wq` 保存并退出 

## 服务器相关
#### 登录注销
`ssh user@hostname` 使用SSH协议登录远程服务器。

`exit 或 logout` 退出当前会话。

#### 网络操作
`ping host` 测试到远程主机的网络连接。

`netstat -an` 显示网络状态。

`ifconfig 或 ip addr` 显示网络接口配置。

#### 进程管理
`nohup python train.py > train.log &` #后台运行进程并存入train.log日志中

`ps -x` 显示当前 所有 进程状态。

`top` 实时显示进程信息。

`htop` 增强版的top命令，以更友好的方式显示进程信息。

`kill PID`例如`kill -9 179757` 发送信号到指定的进程ID。

#### 备份和压缩
`tar` 打包和压缩文件。

`gzip` 压缩文件。

`unzip` 解压文件

`rsync` 同步文件和目录。

#### 权限和用户管理
`chmod` 改变文件或目录权限。

`chown` 改变文件或目录的所有者。

`useradd username` 添加新用户。

`userdel username` 删除用户。

#### 日志管理
`tail -f /var/log/syslog` 实时查看系统日志。

`grep 'pattern' /var/log/apache2/access.log` 在日志文件中搜索特定模式。

