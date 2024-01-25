# README

## 环境
GPU V100，显存16G，如果显存小于16G，需要更改code/configs/ResNet50.yml中的batch_size  
CUDA 9.0  
CUDNN 7.5

## 依赖的软件包
ffmpeg  
sox  
Python >= 3.6  
python包
- catalyst==20.6  
- librosa==0.7.2  
- torch==1.4.0  
- torchvision==0.5.0  

## 解决方案及算法
分数据、网络结构和loss来叙述。  

### 数据
1. 对训练音频数据重采样；
1. 作5秒切片，每个音频会生成几十个训练样本；
1. 变换为梅尔频谱，输出分别以时间和频率为二轴的功率谱，以此作为CNN的输入。

### 网络结构
ResNet50  
定义在code/main.py

### loss
因为本次比较有2个任务，一个是判断真伪，另一个是如果是真的情况下，判断类别。
因此并没有采用通常的softmax + cross-entropy，
而是采用类别个数的BCE-Binary cross-entropy。
当所有loss的最大值低于某threshold时，判断为伪；
高于些threshold时，判断为真，此最大值所对应的index即为输出类别。
用一个loss，同时完成了2个任务。


## 预处理数据
cd code
./main.sh prepare
这个要持续较长时间，视计算机性能，可能要持续1个小时以上。

## 训练
cd code  
./main.sh train # catalyst框架

训练100个epoch，视计算机性能，可能要持续10个小时以上。

## 预测
cd code  
./main.sh test
较快，持续1个小时以内。