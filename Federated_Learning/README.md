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
可以参考文献深度理解联邦学习，为什么要用联邦学习，联邦学习是如何保护数据的

1. Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol., 10(2), 12:11-12:19. https://doi.org/10.1145/3298981 
2. Kairouz, P., McMahan, et al. (2021). Advances and Open Problems in Federated Learning. Found. Trends Mach. Learn., 14(1-2), 1-210. https://doi.org/10.1561/2200000083 
3. Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated Learning: Challenges, Methods, and Future Directions. IEEE Signal Process. Mag., 37(3), 50-60. https://doi.org/10.1109/MSP.2020.2975749

## 文件夹介绍
- [Customization](./Customization)  使用tensorflow实现的 FL 案例
- [office_tutorial](./office_tutorial) 记录自己在官方指南的学习
- [paper_model](./paper_model) 记录文献提出来的：FL Model
- test_fl.py  用于测试在学习中的 api 使用说明
- test_gpu.py 文件，用于检测是否有 gpu, cpu,已经是否可以使用
- TFF_API.md 记录自己在官方学习 TFF_API 指南


