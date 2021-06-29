# AlgorithmEngineerPrepares
Structured **MINIMAL** knowledge system required for an algorithm engineer

**算法工程师的自我修养**【算法最小必要知识体系概略】**（持续更新ing）**

---

### 0x000 [计算机视觉](./0.ComputerVision)

#### 基础概念

[图像分类经典模型（LeNet、AlexNet、VGG、ResNet、GoogLeNet）](./0.ComputerVision/0.Basics/0.ClassBasics.md)

[语义分割任务经典模型（FCN、Unet、RefineNet、PSPnet、SegNet）](./0.ComputerVision/0.Basics/1.SegBasics.md)

[目标检测任务基础模型（两阶段、一阶段，包括R-CNN系列、YOLO系列、SSD）](./0.ComputerVision/0.Basics/2.DetectionBasics.md)

实例分割基本概念与经典模型

计算机视觉中的loss函数

[CNN的轻量化与压缩的基本概念](./0.ComputerVision/0.Basics/5.LiteMethods.md)

蒸馏（distillation）的基本原理；

YOLO实现细节；

DeepLab实现细节；

非极大值抑制方法（NMS）及其实现；

数据增广与测试时增强（TTA）；

low-level任务基本概念；

RoIAlign代码详解

[DeepLab中的CRF的实现](./0.ComputerVision/0.Basics/13.CRFsInCV.md)



#### 低层视觉技术（low-level task）

图像去噪；去雾；超分辨率；去雨雪；



#### 分割与检测

[DeepLab系列模型内容梳理](./0.ComputerVision/2.Seg&Det/0.DeepLab.md)

[并行的多尺度融合策略：HRNet](./0.ComputerVision/2.Seg&Det/1.HRNet.md)

[多次稠密上采样后融合的unet加强版：Unet++](./0.ComputerVision/2.Seg&Det/2.Unet++.md)

[OCRNet（Object Contextural Representation）：基于transformer attention机制的上下文关联策略](./0.ComputerVision/2.Seg&Det/3.OCRNet.md)

[Dilated Residual Net（DRN）：空洞卷积改进resnet](./0.ComputerVision/2.Seg&Det/4.DilatedResNet.md)

#### 生成模型（GAN etc.）

生成对抗网络GAN基本原理；Cycle GAN；Style GAN；Wasserstein GAN；



#### 神经网络鲁棒性与攻击

神经网络的鲁棒性（clever Hans效应）



#### 轻量化模型

[Mobilenet系列模型梳理](./0.ComputerVision/5.LiteModels/0.Moblienet.md)

[Shufflenet v1&v2：分组卷积加通道shuffle信息交互](./0.ComputerVision/5.LiteModels/1.Shufflenet.md)

[Squeezenet：压缩通道学特征与膨胀](./0.ComputerVision/5.LiteModels/2.Squeezenet.md)

[Ghostnet：ghost特征加简单变换生成丰富feature map](./0.ComputerVision/5.LiteModels/3.Ghostnet.md)

[Efficientnet：通道数、深度、分辨率的复合NAS搜索优化](./0.ComputerVision/5.LiteModels/4.EfficientNet.md)



#### 小样本学习

[小样本基本概念](./0.ComputerVision/6.FewShotLearning/0.FewShotBasics.md)

小样本分割经典模型梳理；



#### 弱监督学习

TBD



#### 骨干网络（backbone）

Resnet变体；SENet；DenseNet；Inceptions；



---

### 0x001 [数据处理](./1.DataProcessing)

#### 基础概念

[数据处理和评估中的常用度量指标（AUC/KS/Gain/Lift）](./1.DataProcessing/0.Basics/0.Measurements.md)

特征分箱原则与策略（等频分箱、等距分箱）

[特征分析之单变量分析（WOE/IV/PSI）](./1.DataProcessing/0.Basics/2.SingleFactorAnalysis.md)



---

### 0x002 [深度学习基础 ](./2.DeepLearningBasic)

#### 损失函数

[最常用的损失函数（BCE、交叉熵、MSE、MAE）](./2.DeepLearningBasic/0.Losses/0.CommonLosses.md)

[PyTorch中定义的其他几种损失函数（NLL、Multi-label、Huber、Triplet Loss等）](./2.DeepLearningBasic/0.Losses/1.PytorchLosses.md)

[分割和检测任务中的损失函数（IoU loss、Dice loss、Lovasz loss）](./2.DeepLearningBasic/0.Losses/2.SegDetLosses.md)



#### 下降方法与学习率策略

[梯度下降法及其变体（Momentum、Nesterov、RMSprop、Adagrad、Adam、AdamW等）](./2.DeepLearningBasic/1.Descent&Scheduler/0.GradDescentMethods.md)

[牛顿法和拟牛顿法：二阶优化的策略](./2.DeepLearningBasic/1.Descent&Scheduler/5.NewtonMethod.md)

[Gradient Clipping：梯度裁剪，防止梯度爆炸的手段](./2.DeepLearningBasic/1.Descent&Scheduler/6.GradientClipping.md)

L1 norm下的优化；ISTA&FISTA；学习率策略；ADMM；



#### 激活函数

[常见激活函数（sigmoid/ReLU/Maxout 等）](./2.DeepLearningBasic/2.Activations/0.CommonActivations.md)

[RELU6：适应低精度计算的有界激活函数](./2.DeepLearningBasic/2.Activations/1.ReLU6.md)

[GELU：依据高斯分布cdf计算设计的激活函数](./2.DeepLearningBasic/2.Activations/2.GELU.md)

[Swish：光滑非单调下有界的self-gating函数](./2.DeepLearningBasic/2.Activations/3.Swish.md)



#### 卷积层设计

分组卷积；deformable（可变）卷积



#### 初始化方法

[常用初始化方法（随机初始化、xavier初始化、kaiming初始化）](./2.DeepLearningBasic/4.WeightInitialization/BasicInit.md)



---

### 0x003[（非NN类）机器学习基础](./3.MachineLearningBasic)

#### 必备基础

[SVM、逻辑回归、决策树的关键内容整理](./3.MachineLearningBasic/0.VeryBasic/0.SVM&LR&DT.md)

kNN与k-d树：简单算法的工程优化

[集成学习方法（bagging、boosting、stacking）、梯度提升原理](./3.MachineLearningBasic/0.VeryBasic/2.Ensembles.md)



#### 进阶模型与算法

隐马尔科夫模型HMM；MCMC；Viterbi算法；匈牙利算法；

[宽表数据建模常用集成模型：XGBoost、GBDT、LightGBM，原理与区别](./3.MachineLearningBasic/1.AdvancedMethod/4.Xgboost&GBDT&LightGBM.md)



---

### 0x004 [视觉中的Transformer ](./4.Transformer)

#### 基础概念

注意力机制；



#### 视觉Transformer

ViT模型；Swin Transformer；DETR；SETR；DeiT；CaiT；Segmenter；SegFormer；MLP-Mixer（非transformer）；



---

### 0x005 [自然语言处理基础 ](5.NLPBasic)

#### 必备基础

分词、词频、预处理等；主题模型；



#### 基本NLP模型

[最基础的词嵌入模型Word2vec（skip-gram & cbow）](./5.NLPBasic/1.NLPModels/0.Word2vec.md)

[GloVe词嵌入：基于共现矩阵学习词间**关系**差别的嵌入方法](./5.NLPBasic/1.NLPModels/1.Glove.md)

[ELMo：基于语言模型的bi-LSTM的基于上下文动态嵌入的模型，处理一词多义（芝麻街命名模型第一弹）](./5.NLPBasic/1.NLPModels/2.ELMo.md)

[FastText：Facebook开源的文本分类和词嵌入工具包](./5.NLPBasic/1.NLPModels/3.Fasttext.md)

[GPT系列：Generative Pre-Training，单向Transformer，非监督预训练+下游任务微调](./5.NLPBasic/1.NLPModels/4.GPT.md)

[BERT：谷歌开放的基于bidirectional Transformer的最强语言模型，BERT yyds !](./5.NLPBasic/1.NLPModels/5.BERT&Variants.md)



#### NLP模型实现的相关tricks

[Negative Sampling：词嵌入训练中的负采样操作](./5.NLPBasic/2.NLPTricks/0.NegativeSampling.md)

[Hierachical Softmax：基于Huffman编码的高输出维度（词典大小）的优化方法](./5.NLPBasic/2.NLPTricks/1.HierarchicalSoftmax.md)



---

### 0x006 信息论 [🐶](./6.InformationTheory)

TBD





---

### 0x007 矩阵论 [🐶](./7.Matrix)

TBD





---

### 0x008 统计推断与概率论 [🐶](./8.StatisticalInference)

TBD





---

### 0x009 PyTorch相关 [🐶](./9.PyTorch)

TBD







---

### 0xFFF 常用工具类 [🐶](./999.Tools)

#### Git相关

#### hadoop相关

#### SQL相关

#### Docker相关

#### 后端开发语言

#### 前端开发语言

