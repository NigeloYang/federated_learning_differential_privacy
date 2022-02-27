# 联邦学习
## 如何安装
因为笔者使用的是 Windows 系统学习 Tensorflow Federated Learning 所以适配的时候会出现部分问题（Tensorflow-Federated  > v.0.17.0 以后）。
所以，在这里我们使用最适合 Windows 系统的安装包，在自己的 python 环境中安装如下：
1. 使用 conda 创建一个虚拟环境 conda create -n tff python=3.8
2. 激活环境 conda activate tff
3. 安装 Tensorflow-gpu 版本 :  pip install tensorflow-gpu==2.3.0
4. 安装 Federated learning 框架:  pip install tensorflow_federated==0.17.0
5. 如果想安装其他版本，请参考官网：tensorflow and federated-learning 对应的版本

## 什么是联邦学习
可以参考文献深度理解联邦学习，为什么要用联邦学习

1、Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol., 10(2), 12:11-12:19. https://doi.org/10.1145/3298981 

## 文件夹介绍
- demo 文件夹，此文件是案例学习记录 
- FL_API 文件，用于记录学习中常用的 api
- test_gpu.py 文件，用于检测是否有 gpu and cpu 可以使用


