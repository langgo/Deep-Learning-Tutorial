# 深度学习相关论文 Top 1000（通识/引用排序）

说明：本清单采用“人工经典种子 + Crossref 高被引检索”的方式生成，覆盖深度学习基础、神经网络、CNN/RNN/GAN/Transformer/NLP/CV/强化学习/图学习/扩散模型等方向。Crossref 对 NeurIPS/ICLR/arXiv 经典论文覆盖不稳定，因此对公认核心论文做了人工补入和排序校正。引用数会随数据库变化；“通识高被引/排序参考”不是精确 Google Scholar 数字。

共 1000 篇。

## 1. Attention Is All You Need
- 类型：NLP/语言模型
- 标签：Transformer, Self-Attention, Encoder-Decoder, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：190000）
- 年份：2017
- 作者：Ashish Vaswani, Noam Shazeer, Niki Parmar et al.
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/1706.03762
- 概述：提出 Transformer，用自注意力替代循环/卷积结构，成为现代大模型与NLP系统的基础架构。

## 2. Deep Residual Learning for Image Recognition
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：172342
- 年份：2016
- 作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.90
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 3. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- 类型：NLP/语言模型
- 标签：BERT, Transformer, Pretraining, Language Model
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：120000）
- 年份：2018
- 作者：Jacob Devlin, Ming-Wei Chang, Kenton Lee et al.
- 来源：NAACL / arXiv
- 链接：https://arxiv.org/abs/1810.04805
- 概述：提出双向 Transformer 预训练语言模型，通过 masked LM 和 fine-tuning 推动NLP迁移学习范式。

## 4. ImageNet Classification with Deep Convolutional Neural Networks
- 类型：计算机视觉
- 标签：AlexNet, CNN, ImageNet, Supervised Learning
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：110000）
- 年份：2012
- 作者：Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- 来源：NeurIPS
- 链接：https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- 概述：AlexNet 论文，用大规模CNN在ImageNet上取得突破，点燃现代深度学习浪潮。

## 5. Long Short-Term Memory
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：101000）
- 年份：1997
- 作者：Sepp Hochreiter, Jürgen Schmidhuber
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.1997.9.8.1735
- 概述：提出 LSTM 门控记忆单元，缓解RNN长期依赖和梯度消失问题。

## 6. U-Net: Convolutional Networks for Biomedical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：通识高被引（排序参考：100000）
- 年份：2015
- 作者：Olaf Ronneberger, Philipp Fischer, Thomas Brox
- 来源：MICCAI
- 链接：https://doi.org/10.1007/978-3-319-24574-4_28
- 概述：提出 U-Net 编解码与跳连结构，成为医学图像和通用分割任务的经典架构。

## 7. Adam: A Method for Stochastic Optimization
- 类型：深度学习相关
- 标签：Optimization/Training
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：95000）
- 年份：2014
- 作者：Diederik P. Kingma, Jimmy Ba
- 来源：ICLR / arXiv
- 链接：https://arxiv.org/abs/1412.6980
- 概述：提出 Adam 自适应优化器，成为深度神经网络训练中最常用的优化方法之一。

## 8. Generative Adversarial Nets
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：90000）
- 年份：2014
- 作者：Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza et al.
- 来源：NeurIPS
- 链接：https://papers.nips.cc/paper/5423-generative-adversarial-nets
- 概述：提出 GAN 框架，以生成器和判别器对抗训练推动生成模型发展。

## 9. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- 类型：深度学习相关
- 标签：Normalization
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：85000）
- 年份：2015
- 作者：Sergey Ioffe, Christian Szegedy
- 来源：ICML
- 链接：https://arxiv.org/abs/1502.03167
- 概述：提出批归一化，加速深层网络训练并改善优化稳定性。

## 10. Very Deep Convolutional Networks for Large-Scale Image Recognition
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：通识高被引（排序参考：80000）
- 年份：2014
- 作者：Karen Simonyan, Andrew Zisserman
- 来源：ICLR / arXiv
- 链接：https://arxiv.org/abs/1409.1556
- 概述：VGG 论文展示了使用小卷积核堆叠更深CNN的有效性，成为视觉骨干网络经典。

## 11. Deep learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：75643
- 年份：2015
- 作者：Yann LeCun, Yoshua Bengio, Geoffrey Hinton
- 来源：Nature
- 链接：https://doi.org/10.1038/nature14539
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 12. Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：70000）
- 年份：2014
- 作者：Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky et al.
- 来源：JMLR
- 链接：https://jmlr.org/papers/v15/srivastava14a.html
- 概述：系统提出 Dropout 正则化，降低神经网络过拟合并提升泛化能力。

## 13. Learning representations by back-propagating errors
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：65000）
- 年份：1986
- 作者：David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- 来源：Nature
- 链接：https://doi.org/10.1038/323533a0
- 概述：经典反向传播论文，奠定多层神经网络端到端训练的基础。

## 14. Gradient-based learning applied to document recognition
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Optimization/Training
- 重要性：架构论文
- 引用数：通识高被引（排序参考：60000）
- 年份：1998
- 作者：Yann LeCun, Léon Bottou, Yoshua Bengio et al.
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/5.726791
- 概述：LeNet-5 经典论文，展示CNN在手写文档识别中的端到端训练与应用。

## 15. Going Deeper with Convolutions
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：通识高被引（排序参考：58000）
- 年份：2015
- 作者：Christian Szegedy, Wei Liu, Yangqing Jia et al.
- 来源：CVPR
- 链接：https://doi.org/10.1109/cvpr.2015.7298594
- 概述：提出 GoogLeNet/Inception 模块，用多尺度卷积结构提升CNN计算效率和识别性能。

## 16. Efficient Estimation of Word Representations in Vector Space
- 类型：NLP/语言模型
- 标签：Language Model, Word Embedding
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：50000）
- 年份：2013
- 作者：Tomas Mikolov, Kai Chen, Greg Corrado et al.
- 来源：ICLR Workshop / arXiv
- 链接：https://arxiv.org/abs/1301.3781
- 概述：提出 word2vec 的高效词向量训练方法，推动分布式词表示在NLP中的普及。

## 17. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：通识高被引（排序参考：50000）
- 年份：2015
- 作者：Shaoqing Ren, Kaiming He, Ross Girshick et al.
- 来源：NeurIPS / TPAMI
- 链接：https://arxiv.org/abs/1506.01497
- 概述：提出区域提议网络RPN，使两阶段目标检测进入端到端、近实时训练框架。

## 18. You Only Look Once: Unified, Real-Time Object Detection
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：48000）
- 年份：2016
- 作者：Joseph Redmon, Santosh Divvala, Ross Girshick et al.
- 来源：CVPR
- 链接：https://doi.org/10.1109/cvpr.2016.91
- 概述：提出 YOLO 单阶段检测框架，将目标检测表述为统一回归问题以实现实时检测。

## 19. Sequence to Sequence Learning with Neural Networks
- 类型：NLP/语言模型
- 标签：Language Model, RNN
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：45000）
- 年份：2014
- 作者：Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/1409.3215
- 概述：提出基于编码器—解码器LSTM的seq2seq框架，推动端到端序列建模和机器翻译。

## 20. Distributed Representations of Words and Phrases and their Compositionality
- 类型：NLP/语言模型
- 标签：Language Model, Word Embedding
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：45000）
- 年份：2013
- 作者：Tomas Mikolov, Ilya Sutskever, Kai Chen et al.
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/1310.4546
- 概述：提出负采样等技巧并展示词向量组合语义，是word2vec路线的重要论文。

## 21. A Neural Probabilistic Language Model
- 类型：NLP/语言模型
- 标签：Neural Language Model, Word Embedding, Representation Learning
- 重要性：奠基/方法论文
- 引用数：2675（OpenAlex；通识基础论文）（排序参考：43000）
- 年份：2003
- 作者：Yoshua Bengio, Réjean Ducharme, Pascal Vincent et al.
- 来源：Journal of Machine Learning Research
- 链接：https://doi.org/10.5555/944919.944966
- 概述：提出神经概率语言模型，用分布式词表示和神经网络估计词序列概率，是神经语言模型与词向量路线的重要奠基工作。

## 22. Neural Machine Translation by Jointly Learning to Align and Translate
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：42000）
- 年份：2015
- 作者：Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- 来源：ICLR
- 链接：https://arxiv.org/abs/1409.0473
- 概述：提出可学习对齐的注意力机制，显著改进神经机器翻译并影响后续注意力模型。

## 23. GloVe: Global Vectors for Word Representation
- 类型：NLP/语言模型
- 标签：Language Model, Word Embedding
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：40000）
- 年份：2014
- 作者：Jeffrey Pennington, Richard Socher, Christopher D. Manning
- 来源：EMNLP
- 链接：https://aclanthology.org/D14-1162/
- 概述：提出结合全局共现统计和局部上下文窗口的词向量训练方法。

## 24. SSD: Single Shot MultiBox Detector
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：40000）
- 年份：2016
- 作者：Wei Liu, Dragomir Anguelov, Dumitru Erhan et al.
- 来源：ECCV
- 链接：https://doi.org/10.1007/978-3-319-46448-0_2
- 概述：提出SSD单阶段检测器，用多尺度默认框实现高效目标检测。

## 25. Layer Normalization
- 类型：深度学习相关
- 标签：Attention/Transformer, Normalization
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：35000）
- 年份：2016
- 作者：Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
- 来源：arXiv
- 链接：https://arxiv.org/abs/1607.06450
- 概述：提出层归一化，不依赖batch统计，广泛用于RNN和Transformer训练。

## 26. Densely Connected Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：34044
- 年份：2017
- 作者：Gao Huang, Zhuang Liu, Laurens Van Der Maaten et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.243
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 27. Rectified Linear Units Improve Restricted Boltzmann Machines
- 类型：深度学习相关
- 标签：Optimization/Training, Unsupervised Representation
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：33000）
- 年份：2010
- 作者：Vinod Nair, Geoffrey E. Hinton
- 来源：ICML
- 链接：https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
- 概述：推动ReLU激活函数在深度模型中的应用，改善训练效率和稀疏表征。

## 28. Understanding the difficulty of training deep feedforward neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：通识高被引（排序参考：32000）
- 年份：2010
- 作者：Xavier Glorot, Yoshua Bengio
- 来源：AISTATS
- 链接：http://proceedings.mlr.press/v9/glorot10a.html
- 概述：分析深层前馈网络训练困难并提出Xavier初始化思想。

## 29. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
- 类型：计算机视觉
- 标签：视觉, Optimization/Training, Dataset/Benchmark
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：30000）
- 年份：2015
- 作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren et al.
- 来源：ICCV
- 链接：https://doi.org/10.1109/iccv.2015.123
- 概述：提出PReLU和He初始化，改善极深CNN训练并刷新ImageNet性能。

## 30. Auto-Encoding Variational Bayes
- 类型：生成模型
- 标签：VAE, Variational Inference, Generative AI
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：30000）
- 年份：2013
- 作者：Diederik P. Kingma, Max Welling
- 来源：ICLR
- 链接：https://arxiv.org/abs/1312.6114
- 概述：提出VAE的重参数化技巧和变分自编码框架，成为深度生成模型基础。

## 31. Language Models are Few-Shot Learners
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：30000）
- 年份：2020
- 作者：Tom B. Brown, Benjamin Mann, Nick Ryder et al.
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/2005.14165
- 概述：GPT-3论文，展示大规模语言模型的上下文学习和少样本能力。

## 32. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：29964
- 年份：2021
- 作者：Ze Liu, Yutong Lin, Yue Cao et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00986
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 33. Fully convolutional networks for semantic segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：28305
- 年份：2015
- 作者：Jonathan Long, Evan Shelhamer, Trevor Darrell
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7298965
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 34. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, CNN, GAN
- 重要性：架构论文
- 引用数：通识高被引（排序参考：28000）
- 年份：2015
- 作者：Alec Radford, Luke Metz, Soumith Chintala
- 来源：ICLR
- 链接：https://arxiv.org/abs/1511.06434
- 概述：DCGAN论文，规范化卷积GAN架构并展示无监督视觉表征能力。

## 35. Mask R-CNN
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：26822
- 年份：2017
- 作者：Kaiming He, Georgia Gkioxari, Piotr Dollar et al.
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.322
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 36. Playing Atari with Deep Reinforcement Learning
- 类型：强化学习
- 标签：RL
- 重要性：训练/推理方法
- 引用数：通识高被引（排序参考：26000）
- 年份：2013
- 作者：Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al.
- 来源：NeurIPS Workshop / arXiv
- 链接：https://arxiv.org/abs/1312.5602
- 概述：DQN早期论文，用卷积网络从像素输入学习Atari控制策略。

## 37. Fast R-CNN
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：25304
- 年份：2015
- 作者：Ross Girshick
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.169
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 38. Human-level control through deep reinforcement learning
- 类型：强化学习
- 标签：RL
- 重要性：训练/推理方法
- 引用数：24049
- 年份：2015
- 作者：Volodymyr Mnih, Koray Kavukcuoglu, David Silver et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/nature14236
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 39. Continuous control with deep reinforcement learning
- 类型：强化学习
- 标签：RL
- 重要性：训练/推理方法
- 引用数：通识高被引（排序参考：24000）
- 年份：2015
- 作者：Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel et al.
- 来源：ICLR
- 链接：https://arxiv.org/abs/1509.02971
- 概述：提出DDPG，将深度强化学习扩展到连续动作控制问题。

## 40. RoBERTa: A Robustly Optimized BERT Pretraining Approach
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：23000）
- 年份：2019
- 作者：Yinhan Liu, Myle Ott, Naman Goyal et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/1907.11692
- 概述：系统优化BERT预训练策略，证明训练数据、步数和目标设置对性能影响巨大。

## 41. MobileNetV2: Inverted Residuals and Linear Bottlenecks
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection, CNN
- 重要性：奠基/方法论文
- 引用数：22292
- 年份：2018
- 作者：Mark Sandler, Andrew Howard, Menglong Zhu et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00474
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 42. Rethinking the Inception Architecture for Computer Vision
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：22217
- 年份：2016
- 作者：Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.308
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 43. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer, Optimization/Training
- 重要性：奠基/方法论文
- 引用数：通识高被引（排序参考：22000）
- 年份：2019
- 作者：Colin Raffel, Noam Shazeer, Adam Roberts et al.
- 来源：JMLR
- 链接：https://arxiv.org/abs/1910.10683
- 概述：T5论文，将多种NLP任务统一为text-to-text格式并系统研究迁移学习。

## 44. Wasserstein GAN
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：21000）
- 年份：2017
- 作者：Martin Arjovsky, Soumith Chintala, Léon Bottou
- 来源：ICML
- 链接：https://arxiv.org/abs/1701.07875
- 概述：引入Wasserstein距离改进GAN训练稳定性和损失可解释性。

## 45. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI
- 重要性：架构论文
- 引用数：通识高被引（排序参考：21000）
- 年份：2017
- 作者：Jun-Yan Zhu, Taesung Park, Phillip Isola et al.
- 来源：ICCV
- 链接：https://doi.org/10.1109/iccv.2017.244
- 概述：CycleGAN论文，用循环一致性实现无配对图像到图像翻译。

## 46. Denoising Diffusion Probabilistic Models
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：20000）
- 年份：2020
- 作者：Jonathan Ho, Ajay Jain, Pieter Abbeel
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/2006.11239
- 概述：DDPM论文，用逐步加噪与去噪训练高质量生成模型，推动扩散模型热潮。

## 47. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：应用/方法论文
- 引用数：18411
- 年份：2018
- 作者：Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2017.2699184
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 48. XLNet: Generalized Autoregressive Pretraining for Language Understanding
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：18000）
- 年份：2019
- 作者：Zhilin Yang, Zihang Dai, Yiming Yang et al.
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/1906.08237
- 概述：提出排列语言建模，结合自回归预训练和双向上下文建模。

## 49. Semi-Supervised Classification with Graph Convolutional Networks
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：架构论文
- 引用数：通识高被引（排序参考：18000）
- 年份：2017
- 作者：Thomas N. Kipf, Max Welling
- 来源：ICLR
- 链接：https://arxiv.org/abs/1609.02907
- 概述：提出GCN的简洁谱图卷积形式，成为图神经网络基础模型。

## 50. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：17958
- 年份：2019
- 作者：M. Raissi, P. Perdikaris, G.E. Karniadakis
- 来源：Journal of Computational Physics
- 链接：https://doi.org/10.1016/j.jcp.2018.10.045
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 51. Reducing the Dimensionality of Data with Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：17123
- 年份：2006
- 作者：G. E. Hinton, R. R. Salakhutdinov
- 来源：Science
- 链接：https://doi.org/10.1126/science.1127647
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 52. PyTorch: An Imperative Style, High-Performance Deep Learning Library
- 类型：系统/框架
- 标签：Framework/System, Optimization/Training
- 重要性：系统/框架
- 引用数：通识高被引（排序参考：17000）
- 年份：2019
- 作者：Adam Paszke, Sam Gross, Francisco Massa et al.
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/1912.01703
- 概述：介绍PyTorch动态图深度学习框架，强调易用性和高性能训练。

## 53. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
- 类型：NLP/语言模型
- 标签：Language Model, RNN
- 重要性：架构论文
- 引用数：16233
- 年份：2014
- 作者：Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre et al.
- 来源：Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)
- 链接：https://doi.org/10.3115/v1/d14-1179
- 概述：研究序列到序列/神经机器翻译模型，推动端到端NLP建模。

## 54. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer, CNN, Segmentation
- 重要性：架构论文
- 引用数：16210
- 年份：2017
- 作者：Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2016.2644615
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 55. Proximal Policy Optimization Algorithms
- 类型：强化学习
- 标签：RL, Optimization/Training
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：16000）
- 年份：2017
- 作者：John Schulman, Filip Wolski, Prafulla Dhariwal et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/1707.06347
- 概述：提出PPO策略优化算法，以简单稳定的剪切目标成为强化学习常用基线。

## 56. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：通识高被引（排序参考：16000）
- 年份：2015
- 作者：Martín Abadi, Ashish Agarwal, Paul Barham et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/1603.04467
- 概述：介绍TensorFlow系统，用数据流图支持异构分布式机器学习训练和部署。

## 57. Image-to-Image Translation with Conditional Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI
- 重要性：架构论文
- 引用数：通识高被引（排序参考：16000）
- 年份：2017
- 作者：Phillip Isola, Jun-Yan Zhu, Tinghui Zhou et al.
- 来源：CVPR
- 链接：https://doi.org/10.1109/cvpr.2017.632
- 概述：pix2pix论文，用条件GAN实现成对图像到图像翻译。

## 58. High-Resolution Image Synthesis with Latent Diffusion Models
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：15547
- 年份：2022
- 作者：Robin Rombach, Andreas Blattmann, Dominik Lorenz et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01042
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 59. Asynchronous Methods for Deep Reinforcement Learning
- 类型：强化学习
- 标签：RL
- 重要性：训练/推理方法
- 引用数：通识高被引（排序参考：15000）
- 年份：2016
- 作者：Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza et al.
- 来源：ICML
- 链接：https://arxiv.org/abs/1602.01783
- 概述：提出A3C等异步深度强化学习方法，提高训练效率和稳定性。

## 60. Graph Attention Networks
- 类型：图学习
- 标签：Graph, Attention/Transformer
- 重要性：架构论文
- 引用数：通识高被引（排序参考：15000）
- 年份：2018
- 作者：Petar Veličković, Guillem Cucurull, Arantxa Casanova et al.
- 来源：ICLR
- 链接：https://arxiv.org/abs/1710.10903
- 概述：提出GAT，在图结构邻域聚合中引入注意力权重。

## 61. Xception: Deep Learning with Depthwise Separable Convolutions
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：应用/方法论文
- 引用数：14968
- 年份：2017
- 作者：Francois Chollet
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.195
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 62. Neural networks and physical systems with emergent collective computational abilities.
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：14700
- 年份：1982
- 作者：J J Hopfield
- 来源：Proceedings of the National Academy of Sciences
- 链接：https://doi.org/10.1073/pnas.79.8.2554
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 63. Deep learning in neural networks: An overview
- 类型：深度学习相关
- 标签：Segmentation
- 重要性：综述论文
- 引用数：14607
- 年份：2015
- 作者：Jürgen Schmidhuber
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2014.09.003
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 64. Generative adversarial networks
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：架构论文
- 引用数：14074
- 年份：2020
- 作者：Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza et al.
- 来源：Communications of the ACM
- 链接：https://doi.org/10.1145/3422622
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 65. Inductive Representation Learning on Large Graphs
- 类型：图学习
- 标签：Graph
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：14000）
- 年份：2017
- 作者：William L. Hamilton, Rex Ying, Jure Leskovec
- 来源：NeurIPS
- 链接：https://arxiv.org/abs/1706.02216
- 概述：GraphSAGE论文，通过邻域采样和聚合实现大图归纳式节点表征学习。

## 66. Squeeze-and-Excitation Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：通识高被引（排序参考：14000）
- 年份：2018
- 作者：Jie Hu, Li Shen, Gang Sun
- 来源：CVPR
- 链接：https://doi.org/10.1109/cvpr.2018.00745
- 概述：提出SE通道注意力模块，显著提升CNN特征重标定能力。

## 67. A survey on deep learning in medical image analysis
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：综述论文
- 引用数：13149
- 年份：2017
- 作者：Geert Litjens, Thijs Kooi, Babak Ehteshami Bejnordi et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2017.07.005
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 68. A Fast Learning Algorithm for Deep Belief Nets
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：13056
- 年份：2006
- 作者：Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.2006.18.7.1527
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 69. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：12631
- 年份：2018
- 作者：Richard Zhang, Phillip Isola, Alexei A. Efros et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00068
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 70. Dermatologist-level classification of skin cancer with deep neural networks
- 类型：NLP/语言模型 / 医疗/生命科学AI
- 标签：Language Model, Medical AI, Attention/Transformer
- 重要性：架构论文
- 引用数：12323
- 年份：2017
- 作者：Andre Esteva, Brett Kuprel, Roberto A. Novoa et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/nature21056
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 71. Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：通识高被引（排序参考：12000）
- 年份：2015
- 作者：Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan et al.
- 来源：ICML
- 链接：https://arxiv.org/abs/1503.03585
- 概述：早期扩散生成模型论文，将非平衡热力学过程用于逐步生成建模。

## 72. Mastering the game of Go with deep neural networks and tree search
- 类型：强化学习
- 标签：RL
- 重要性：架构论文
- 引用数：11477
- 年份：2016
- 作者：David Silver, Aja Huang, Chris J. Maddison et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/nature16961
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 73. A survey on Image Data Augmentation for Deep Learning
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：11313
- 年份：2019
- 作者：Connor Shorten, Taghi M. Khoshgoftaar
- 来源：Journal of Big Data
- 链接：https://doi.org/10.1186/s40537-019-0197-0
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 74. Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：9884
- 年份：2015
- 作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2015.2389824
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 75. A Comprehensive Survey on Graph Neural Networks
- 类型：图学习 / 系统/框架
- 标签：Graph, Framework/System
- 重要性：综述论文
- 引用数：9655
- 年份：2021
- 作者：Zonghan Wu, Shirui Pan, Fengwen Chen et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2020.2978386
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 76. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：架构论文
- 引用数：9488
- 年份：2021
- 作者：Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl et al.
- 来源：Nature Methods
- 链接：https://doi.org/10.1038/s41592-020-01008-z
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 77. Backpropagation Applied to Handwritten Zip Code Recognition
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：9366
- 年份：1989
- 作者：Y. LeCun, B. Boser, J. S. Denker et al.
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.1989.1.4.541
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 78. Non-local Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：8980
- 年份：2018
- 作者：Xiaolong Wang, Ross Girshick, Abhinav Gupta et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00813
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 79. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- 类型：NLP/语言模型
- 标签：BERT, Transformer, Pretraining, Language Model
- 重要性：奠基/方法论文
- 引用数：8970
- 年份：2019
- 作者：Nils Reimers, Iryna Gurevych
- 来源：Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)
- 链接：https://doi.org/10.18653/v1/d19-1410
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 80. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：8963
- 年份：2017
- 作者：Christian Ledig, Lucas Theis, Ferenc Huszar et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.19
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 81. Convolutional Neural Networks for Sentence Classification
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, CNN
- 重要性：架构论文
- 引用数：8903
- 年份：2014
- 作者：Yoon Kim
- 来源：Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)
- 链接：https://doi.org/10.3115/v1/d14-1181
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 82. Image Super-Resolution Using Deep Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：8829
- 年份：2016
- 作者：Chao Dong, Chen Change Loy, Kaiming He et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2015.2439281
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 83. Aggregated Residual Transformations for Deep Neural Networks
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：8798
- 年份：2017
- 作者：Saining Xie, Ross Girshick, Piotr Dollar et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.634
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 84. Searching for MobileNetV3
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：应用/方法论文
- 引用数：8765
- 年份：2019
- 作者：Andrew Howard, Mark Sandler, Bo Chen et al.
- 来源：2019 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2019.00140
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 85. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：8560
- 年份：2016
- 作者：Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi
- 来源：2016 Fourth International Conference on 3D Vision (3DV)
- 链接：https://doi.org/10.1109/3dv.2016.79
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 86. Caffe
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：8456
- 年份：2014
- 作者：Yangqing Jia, Evan Shelhamer, Jeff Donahue et al.
- 来源：Proceedings of the 22nd ACM international conference on Multimedia
- 链接：https://doi.org/10.1145/2647868.2654889
- 概述：介绍深度学习框架或系统，支撑神经网络模型的训练、部署与研究复现。

## 87. A Style-Based Generator Architecture for Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：8292
- 年份：2019
- 作者：Tero Karras, Samuli Laine, Timo Aila
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00453
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 88. Learning Deep Features for Discriminative Localization
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：8196
- 年份：2016
- 作者：Bolei Zhou, Aditya Khosla, Agata Lapedriza et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.319
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 89. Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection, CNN
- 重要性：奠基/方法论文
- 引用数：8155
- 年份：2017
- 作者：Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v31i1.11231
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 90. Bidirectional recurrent neural networks
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：8071
- 年份：1997
- 作者：M. Schuster, K.K. Paliwal
- 来源：IEEE Transactions on Signal Processing
- 链接：https://doi.org/10.1109/78.650093
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 91. AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models
- 类型：医疗/生命科学AI
- 标签：Medical AI, Dataset/Benchmark
- 重要性：数据集/基准
- 引用数：7962
- 年份：2022
- 作者：Mihaly Varadi, Stephen Anyango, Mandar Deshpande et al.
- 来源：Nucleic Acids Research
- 链接：https://doi.org/10.1093/nar/gkab1061
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 92. Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：7942
- 年份：2012
- 作者：Geoffrey Hinton, Li Deng, Dong Yu et al.
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/msp.2012.2205597
- 概述：研究深度语音/音频建模，服务于识别、分离或表征学习任务。

## 93. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：7889
- 年份：2020
- 作者：Qilong Wang, Banggu Wu, Pengfei Zhu et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.01155
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 94. Brownian motion in a field of force and the diffusion model of chemical reactions
- 类型：生成模型
- 标签：Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：7859
- 年份：1940
- 作者：H.A. Kramers
- 来源：Physica
- 链接：https://doi.org/10.1016/s0031-8914(40)90098-2
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 95. The Graph Neural Network Model
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：7843
- 年份：2009
- 作者：F. Scarselli, M. Gori, Ah Chung Tsoi et al.
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/tnn.2008.2005605
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 96. Design and synthesis of an exceptionally stable and highly porous metal-organic framework
- 类型：计算机视觉 / 生成模型 / 系统/框架
- 标签：视觉, Generative AI, Framework/System, CNN
- 重要性：系统/框架
- 引用数：7810
- 年份：1999
- 作者：Hailian Li, Mohamed Eddaoudi, M. O'Keeffe et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/46248
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 97. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection, CNN
- 重要性：奠基/方法论文
- 引用数：7782
- 年份：2017
- 作者：Kai Zhang, Wangmeng Zuo, Yunjin Chen et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2017.2662206
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 98. DeepWalk
- 类型：图学习
- 标签：Graph
- 重要性：应用/方法论文
- 引用数：7768
- 年份：2014
- 作者：Bryan Perozzi, Rami Al-Rfou, Steven Skiena
- 来源：Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining
- 链接：https://doi.org/10.1145/2623330.2623732
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 99. Learning Spatiotemporal Features with 3D Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：7343
- 年份：2015
- 作者：Du Tran, Lubomir Bourdev, Rob Fergus et al.
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.510
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 100. Masked Autoencoders Are Scalable Vision Learners
- 类型：计算机视觉
- 标签：视觉, Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：7253
- 年份：2022
- 作者：Kaiming He, Xinlei Chen, Saining Xie et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01553
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 101. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：7189
- 年份：2018
- 作者：Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00716
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 102. Review of deep learning: concepts, CNN architectures, challenges, applications, future directions
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：综述论文
- 引用数：6933
- 年份：2021
- 作者：Laith Alzubaidi, Jinglan Zhang, Amjad J. Humaidi et al.
- 来源：Journal of Big Data
- 链接：https://doi.org/10.1186/s40537-021-00444-8
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 103. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：6654
- 年份：2016
- 作者：Varun Gulshan, Lily Peng, Marc Coram et al.
- 来源：JAMA
- 链接：https://doi.org/10.1001/jama.2016.17216
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 104. Enhanced Deep Residual Networks for Single Image Super-Resolution
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：6584
- 年份：2017
- 作者：Bee Lim, Sanghyun Son, Heewon Kim et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- 链接：https://doi.org/10.1109/cvprw.2017.151
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 105. Identification and control of dynamical systems using neural networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：6475
- 年份：1990
- 作者：K.S. Narendra, K. Parthasarathy
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.80202
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 106. Overcoming catastrophic forgetting in neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：6071
- 年份：2017
- 作者：James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz et al.
- 来源：Proceedings of the National Academy of Sciences
- 链接：https://doi.org/10.1073/pnas.1611835114
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 107. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：6039
- 年份：2021
- 作者：Haoyi Zhou, Shanghang Zhang, Jieqi Peng et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v35i12.17325
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 108. Deformable Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：6007
- 年份：2017
- 作者：Jifeng Dai, Haozhi Qi, Yuwen Xiong et al.
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.89
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 109. Learning Deep Architectures for AI
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：5950
- 年份：2009
- 作者：Y. Bengio
- 来源：Foundations and Trends® in Machine Learning
- 链接：https://doi.org/10.1561/2200000006
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 110. Cascade R-CNN: Delving Into High Quality Object Detection
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：5928
- 年份：2018
- 作者：Zhaowei Cai, Nuno Vasconcelos
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00644
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 111. Accurate Image Super-Resolution Using Very Deep Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：5803
- 年份：2016
- 作者：Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.182
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 112. Towards Evaluating the Robustness of Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：5792
- 年份：2017
- 作者：Nicholas Carlini, David Wagner
- 来源：2017 IEEE Symposium on Security and Privacy (SP)
- 链接：https://doi.org/10.1109/sp.2017.49
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 113. Speech recognition with deep recurrent neural networks
- 类型：语音/音频
- 标签：Speech, RNN
- 重要性：架构论文
- 引用数：5740
- 年份：2013
- 作者：Alex Graves, Abdel-rahman Mohamed, Geoffrey Hinton
- 来源：2013 IEEE International Conference on Acoustics, Speech and Signal Processing
- 链接：https://doi.org/10.1109/icassp.2013.6638947
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 114. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
- 类型：计算机视觉 / 视频/世界模型
- 标签：视觉, Video, CNN
- 重要性：架构论文
- 引用数：5736
- 年份：2016
- 作者：Wenzhe Shi, Jose Caballero, Ferenc Huszar et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.207
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 115. Dual Attention Network for Scene Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：5735
- 年份：2019
- 作者：Jun Fu, Jing Liu, Haijie Tian et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00326
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 116. ChatGPT for good? On opportunities and challenges of large language models for education
- 类型：NLP/语言模型 / 强化学习
- 标签：Language Model, RL, Attention/Transformer
- 重要性：架构论文
- 引用数：5689
- 年份：2023
- 作者：Enkelejda Kasneci, Kathrin Sessler, Stefan Küchemann et al.
- 来源：Learning and Individual Differences
- 链接：https://doi.org/10.1016/j.lindif.2023.102274
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 117. ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：5650
- 年份：2019
- 作者：Jiankang Deng, Jia Guo, Niannan Xue et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00482
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 118. Neural Collaborative Filtering
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：5634
- 年份：2017
- 作者：Xiangnan He, Lizi Liao, Hanwang Zhang et al.
- 来源：Proceedings of the 26th International Conference on World Wide Web
- 链接：https://doi.org/10.1145/3038912.3052569
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 119. Accurate prediction of protein structures and interactions using a three-track neural network
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：5577
- 年份：2021
- 作者：Minkyung Baek, Frank DiMaio, Ivan Anishchenko et al.
- 来源：Science
- 链接：https://doi.org/10.1126/science.abj8754
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 120. LSTM: A Search Space Odyssey
- 类型：系统/框架
- 标签：Framework/System, RNN
- 重要性：系统/框架
- 引用数：5565
- 年份：2017
- 作者：Klaus Greff, Rupesh K. Srivastava, Jan Koutnik et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2016.2582924
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 121. Recent advances in convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：5554
- 年份：2018
- 作者：Jiuxiang Gu, Zhenhua Wang, Jason Kuen et al.
- 来源：Pattern Recognition
- 链接：https://doi.org/10.1016/j.patcog.2017.10.013
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 122. Evolutionary-scale prediction of atomic-level protein structure with a language model
- 类型：NLP/语言模型 / 医疗/生命科学AI
- 标签：Language Model, Medical AI
- 重要性：应用/方法论文
- 引用数：5522
- 年份：2023
- 作者：Zeming Lin, Halil Akin, Roshan Rao et al.
- 来源：Science
- 链接：https://doi.org/10.1126/science.ade2574
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 123. Deep Learning Face Attributes in the Wild
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：5441
- 年份：2015
- 作者：Ziwei Liu, Ping Luo, Xiaogang Wang et al.
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.425
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 124. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：应用/方法论文
- 引用数：5408
- 年份：2018
- 作者：Alexander Mathis, Pranav Mamidanna, Kevin M. Cury et al.
- 来源：Nature Neuroscience
- 链接：https://doi.org/10.1038/s41593-018-0209-y
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 125. A Review of Recurrent Neural Networks: LSTM Cells and Network Architectures
- 类型：深度学习相关
- 标签：RNN
- 重要性：综述论文
- 引用数：5274
- 年份：2019
- 作者：Yong Yu, Xiaosheng Si, Changhua Hu et al.
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco_a_01199
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 126. Graph neural networks: A review of methods and applications
- 类型：图学习
- 标签：Graph
- 重要性：综述论文
- 引用数：5257
- 年份：2020
- 作者：Jie Zhou, Ganqu Cui, Shengding Hu et al.
- 来源：AI Open
- 链接：https://doi.org/10.1016/j.aiopen.2021.01.001
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 127. 3D Convolutional Neural Networks for Human Action Recognition
- 类型：计算机视觉 / 视频/世界模型
- 标签：视觉, Video, CNN
- 重要性：架构论文
- 引用数：5243
- 年份：2013
- 作者：Shuiwang Ji, Wei Xu, Ming Yang et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2012.59
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 128. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：5216
- 年份：2016
- 作者：Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li et al.
- 来源：IEEE Signal Processing Letters
- 链接：https://doi.org/10.1109/lsp.2016.2603342
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 129. Emerging Properties in Self-Supervised Vision Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Self-supervised Learning
- 重要性：架构论文
- 引用数：5214
- 年份：2021
- 作者：Mathilde Caron, Hugo Touvron, Ishan Misra et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00951
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 130. Dynamic Graph CNN for Learning on Point Clouds
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：5191
- 年份：2019
- 作者：Yue Wang, Yongbin Sun, Ziwei Liu et al.
- 来源：ACM Transactions on Graphics
- 链接：https://doi.org/10.1145/3326362
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 131. Transformers: State-of-the-Art Natural Language Processing
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System, Attention/Transformer
- 重要性：系统/框架
- 引用数：5030
- 年份：2020
- 作者：Thomas Wolf, Lysandre Debut, Victor Sanh et al.
- 来源：Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations
- 链接：https://doi.org/10.18653/v1/2020.emnlp-demos.6
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 132. Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：4994
- 年份：2020
- 作者：Alex Sherstinsky
- 来源：Physica D: Nonlinear Phenomena
- 链接：https://doi.org/10.1016/j.physd.2019.132306
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 133. Deep Convolutional Neural Networks for Computer-Aided Detection: CNN Architectures, Dataset Characteristics and Transfer Learning
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Dataset/Benchmark
- 重要性：数据集/基准
- 引用数：4978
- 年份：2016
- 作者：Hoo-Chang Shin, Holger R. Roth, Mingchen Gao et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2528162
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 134. Extracting and composing robust features with denoising autoencoders
- 类型：计算机视觉
- 标签：视觉, Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：4913
- 年份：2008
- 作者：Pascal Vincent, Hugo Larochelle, Yoshua Bengio et al.
- 来源：Proceedings of the 25th international conference on Machine learning - ICML '08
- 链接：https://doi.org/10.1145/1390156.1390294
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 135. Deep Learning with Differential Privacy
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：4758
- 年份：2016
- 作者：Martin Abadi, Andy Chu, Ian Goodfellow et al.
- 来源：Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security
- 链接：https://doi.org/10.1145/2976749.2978318
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 136. Large-Scale Video Classification with Convolutional Neural Networks
- 类型：计算机视觉 / 视频/世界模型
- 标签：视觉, Video, CNN
- 重要性：架构论文
- 引用数：4748
- 年份：2014
- 作者：Andrej Karpathy, George Toderici, Sanketh Shetty et al.
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2014.223
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 137. Framewise phoneme classification with bidirectional LSTM and other neural network architectures
- 类型：深度学习相关
- 标签：RNN, Segmentation
- 重要性：奠基/方法论文
- 引用数：4741
- 年份：2005
- 作者：Alex Graves, Jürgen Schmidhuber
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2005.06.042
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 138. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：4644
- 年份：2018
- 作者：Vernon J Lawhern, Amelia J Solon, Nicholas R Waytowich et al.
- 来源：Journal of Neural Engineering
- 链接：https://doi.org/10.1088/1741-2552/aace8c
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 139. Object Detection With Deep Learning: A Review
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, Object Detection
- 重要性：综述论文
- 引用数：4618
- 年份：2019
- 作者：Zhong-Qiu Zhao, Peng Zheng, Shou-Tao Xu et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2018.2876865
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 140. Learning to Forget: Continual Prediction with LSTM
- 类型：深度学习相关
- 标签：RNN
- 重要性：奠基/方法论文
- 引用数：4609
- 年份：2000
- 作者：Felix A. Gers, Jürgen Schmidhuber, Fred Cummins
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/089976600300015015
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 141. starBase v2.0: decoding miRNA-ceRNA, miRNA-ncRNA and protein–RNA interaction networks from large-scale CLIP-Seq data
- 类型：医疗/生命科学AI / 多模态
- 标签：Medical AI, Multimodal
- 重要性：架构论文
- 引用数：4605
- 年份：2014
- 作者：Jun-Hao Li, Shun Liu, Hui Zhou et al.
- 来源：Nucleic Acids Research
- 链接：https://doi.org/10.1093/nar/gkt1248
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 142. Deep learning and process understanding for data-driven Earth system science
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：4592
- 年份：2019
- 作者：Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-019-0912-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 143. A guide to deep learning in healthcare
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：4582
- 年份：2019
- 作者：Andre Esteva, Alexandre Robicquet, Bharath Ramsundar et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-018-0316-z
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 144. Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：4521
- 年份：2021
- 作者：Wenhai Wang, Enze Xie, Xiang Li et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00061
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 145. Deep learning in agriculture: A survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：4515
- 年份：2018
- 作者：Andreas Kamilaris, Francesc X. Prenafeta-Boldú
- 来源：Computers and Electronics in Agriculture
- 链接：https://doi.org/10.1016/j.compag.2018.02.016
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 146. Adding Conditional Control to Text-to-Image Diffusion Models
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：4484
- 年份：2023
- 作者：Lvmin Zhang, Anyi Rao, Maneesh Agrawala
- 来源：2023 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv51070.2023.00355
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 147. Image Style Transfer Using Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：4471
- 年份：2016
- 作者：Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.265
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 148. A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：综述论文
- 引用数：4434
- 年份：2022
- 作者：Zewen Li, Fan Liu, Wenjie Yang et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2021.3084827
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 149. Deep Reinforcement Learning with Double Q-Learning
- 类型：强化学习
- 标签：RL
- 重要性：训练/推理方法
- 引用数：4413
- 年份：2016
- 作者：Hado Van Hasselt, Arthur Guez, David Silver
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v30i1.10295
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 150. Effective Approaches to Attention-based Neural Machine Translation
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：4399
- 年份：2015
- 作者：Thang Luong, Hieu Pham, Christopher D. Manning
- 来源：Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing
- 链接：https://doi.org/10.18653/v1/d15-1166
- 概述：研究序列到序列/神经机器翻译模型，推动端到端NLP建模。

## 151. SwinIR: Image Restoration Using Swin Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：4386
- 年份：2021
- 作者：Jingyun Liang, Jiezhang Cao, Guolei Sun et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)
- 链接：https://doi.org/10.1109/iccvw54120.2021.00210
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 152. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：4307
- 年份：1980
- 作者：Kunihiko Fukushima
- 来源：Biological Cybernetics
- 链接：https://doi.org/10.1007/bf00344251
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 153. On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：4304
- 年份：2014
- 作者：Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau et al.
- 来源：Proceedings of SSST-8, Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation
- 链接：https://doi.org/10.3115/v1/w14-4012
- 概述：研究序列到序列/神经机器翻译模型，推动端到端NLP建模。

## 154. Using Deep Learning for Image-Based Plant Disease Detection
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：4249
- 年份：2016
- 作者：Sharada P. Mohanty, David P. Hughes, Marcel Salathé
- 来源：Frontiers in Plant Science
- 链接：https://doi.org/10.3389/fpls.2016.01419
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 155. Deep Learning in Medical Image Analysis
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：综述论文
- 引用数：4223
- 年份：2017
- 作者：Dinggang Shen, Guorong Wu, Heung-Il Suk
- 来源：Annual Review of Biomedical Engineering
- 链接：https://doi.org/10.1146/annurev-bioeng-071516-044442
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 156. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：4212
- 年份：2018
- 作者：Daniel S. Kermany, Michael Goldbaum, Wenjia Cai et al.
- 来源：Cell
- 链接：https://doi.org/10.1016/j.cell.2018.02.010
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 157. A Survey on Vision Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：综述论文
- 引用数：4183
- 年份：2023
- 作者：Kai Han, Yunhe Wang, Hanting Chen et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2022.3152247
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 158. CSPNet: A New Backbone that can Enhance Learning Capability of CNN
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：4174
- 年份：2020
- 作者：Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- 链接：https://doi.org/10.1109/cvprw50498.2020.00203
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 159. SignalP 5.0 improves signal peptide predictions using deep neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：4146
- 年份：2019
- 作者：José Juan Almagro Armenteros, Konstantinos D. Tsirigos, Casper Kaae Sønderby et al.
- 来源：Nature Biotechnology
- 链接：https://doi.org/10.1038/s41587-019-0036-z
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 160. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
- 类型：计算机视觉 / 视频/世界模型 / 图学习
- 标签：视觉, Video, Graph, CNN
- 重要性：架构论文
- 引用数：4058
- 年份：2018
- 作者：Sijie Yan, Yuanjun Xiong, Dahua Lin
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v32i1.12328
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 161. Cellular neural networks: theory
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：4038
- 年份：1988
- 作者：L.O. Chua, L. Yang
- 来源：IEEE Transactions on Circuits and Systems
- 链接：https://doi.org/10.1109/31.7600
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 162. Generative Adversarial Networks: An Overview
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：综述论文
- 引用数：4021
- 年份：2018
- 作者：Antonia Creswell, Tom White, Vincent Dumoulin et al.
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/msp.2017.2765202
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 163. Deep Reinforcement Learning: A Brief Survey
- 类型：强化学习
- 标签：RL
- 重要性：综述论文
- 引用数：4010
- 年份：2017
- 作者：Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage et al.
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/msp.2017.2743240
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 164. Convolutional neural networks: an overview and application in radiology
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：综述论文
- 引用数：3995
- 年份：2018
- 作者：Rikiya Yamashita, Mizuho Nishio, Richard Kinh Gian Do et al.
- 来源：Insights into Imaging
- 链接：https://doi.org/10.1007/s13244-018-0639-9
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 165. A general regression neural network
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3794
- 年份：1991
- 作者：D.F. Specht
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.97934
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 166. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：应用/方法论文
- 引用数：3766
- 年份：2017
- 作者：R. Qi Charles, Hao Su, Mo Kaichun et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.16
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 167. Restormer: Efficient Transformer for High-Resolution Image Restoration
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：3765
- 年份：2022
- 作者：Syed Waqas Zamir, Aditya Arora, Salman Khan et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00564
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 168. Least Squares Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：3696
- 年份：2017
- 作者：Xudong Mao, Qing Li, Haoran Xie et al.
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.304
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 169. Backpropagation through time: what it does and how to do it
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：3658
- 年份：1990
- 作者：P.J. Werbos
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/5.58337
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 170. Large language models in medicine
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：3646
- 年份：2023
- 作者：Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-023-02448-8
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 171. Performance of ChatGPT on USMLE: Potential for AI-assisted medical education using large language models
- 类型：NLP/语言模型 / 医疗/生命科学AI
- 标签：Language Model, Medical AI, Attention/Transformer
- 重要性：架构论文
- 引用数：3568
- 年份：2023
- 作者：Tiffany H. Kung, Morgan Cheatham, Arielle Medenilla et al.
- 来源：PLOS Digital Health
- 链接：https://doi.org/10.1371/journal.pdig.0000198
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 172. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
- 类型：计算机视觉 / 图学习 / 系统/框架
- 标签：视觉, Graph, Framework/System, CNN
- 重要性：系统/框架
- 引用数：3561
- 年份：2018
- 作者：Bing Yu, Haoteng Yin, Zhanxing Zhu
- 来源：Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence
- 链接：https://doi.org/10.24963/ijcai.2018/505
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 173. Time series forecasting using a hybrid ARIMA and neural network model
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3561
- 年份：2003
- 作者：G.Peter Zhang
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/s0925-2312(01)00702-0
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 174. Efficient Processing of Deep Neural Networks: A Tutorial and Survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：3557
- 年份：2017
- 作者：Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang et al.
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/jproc.2017.2761740
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 175. Large language models encode clinical knowledge
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：3541
- 年份：2023
- 作者：Karan Singhal, Shekoofeh Azizi, Tao Tu et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-023-06291-2
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 176. DIA-NN: neural networks and interference correction enable deep proteome coverage in high throughput
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3517
- 年份：2020
- 作者：Vadim Demichev, Christoph B. Messner, Spyros I. Vernardis et al.
- 来源：Nature Methods
- 链接：https://doi.org/10.1038/s41592-019-0638-x
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 177. Deep learning with convolutional neural networks for EEG decoding and visualization
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：3463
- 年份：2017
- 作者：Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef Fiederer et al.
- 来源：Human Brain Mapping
- 链接：https://doi.org/10.1002/hbm.23730
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 178. DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：3375
- 年份：2016
- 作者：Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.282
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 179. Forecasting with artificial neural networks:
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3299
- 年份：1998
- 作者：Guoqiang Zhang, B. Eddy Patuwo, Michael Y. Hu
- 来源：International Journal of Forecasting
- 链接：https://doi.org/10.1016/s0169-2070(97)00044-7
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 180. Neural network ensembles
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3288
- 年份：1990
- 作者：L.K. Hansen, P. Salamon
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/34.58871
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 181. On the approximate realization of continuous mappings by neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3268
- 年份：1989
- 作者：Ken-Ichi Funahashi
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/0893-6080(89)90003-8
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 182. Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：3254
- 年份：2017
- 作者：Xiao Xiang Zhu, Devis Tuia, Lichao Mou et al.
- 来源：IEEE Geoscience and Remote Sensing Magazine
- 链接：https://doi.org/10.1109/mgrs.2017.2762307
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 183. FlowNet: Learning Optical Flow with Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：3243
- 年份：2015
- 作者：Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg et al.
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.316
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 184. A review on the attention mechanism of deep learning
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：综述论文
- 引用数：3240
- 年份：2021
- 作者：Zhaoyang Niu, Guoqiang Zhong, Hui Yu
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2021.03.091
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 185. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：3235
- 年份：2018
- 作者：Benoit Jacob, Skirmantas Kligys, Bo Chen et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00286
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 186. Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：3230
- 年份：2021
- 作者：Sixiao Zheng, Jiachen Lu, Hengshuang Zhao et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00681
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 187. A Learning Algorithm for Continually Running Fully Recurrent Neural Networks
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：3188
- 年份：1989
- 作者：Ronald J. Williams, David Zipser
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.1989.1.2.270
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 188. Performance of neural network basecalling tools for Oxford Nanopore sequencing
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3156
- 年份：2019
- 作者：Ryan R. Wick, Louise M. Judd, Kathryn E. Holt
- 来源：Genome Biology
- 链接：https://doi.org/10.1186/s13059-019-1727-y
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 189. Improved protein structure prediction using potentials from deep learning
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：3113
- 年份：2020
- 作者：Andrew W. Senior, Richard Evans, John Jumper et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-019-1923-7
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 190. StarGAN: Unified Generative Adversarial Networks for Multi-domain Image-to-Image Translation
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：3111
- 年份：2018
- 作者：Yunjey Choi, Minje Choi, Munyoung Kim et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00916
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 191. Masked-attention Mask Transformer for Universal Image Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：3097
- 年份：2022
- 作者：Bowen Cheng, Ishan Misra, Alexander G. Schwing et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00135
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 192. Residual Attention Network for Image Classification
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：3096
- 年份：2017
- 作者：Fei Wang, Mengqing Jiang, Chen Qian et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.683
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 193. Transformers in Vision: A Survey
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：综述论文
- 引用数：3069
- 年份：2022
- 作者：Salman Khan, Muzammal Naseer, Munawar Hayat et al.
- 来源：ACM Computing Surveys
- 链接：https://doi.org/10.1145/3505244
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 194. Deep Convolutional Neural Networks for Image Classification: A Comprehensive Review
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：综述论文
- 引用数：3066
- 年份：2017
- 作者：Waseem Rawat, Zenghui Wang
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco_a_00990
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 195. The solution-diffusion model: a review
- 类型：生成模型
- 标签：Generative AI, Diffusion/Flow
- 重要性：综述论文
- 引用数：3051
- 年份：1995
- 作者：J.G. Wijmans, R.W. Baker
- 来源：Journal of Membrane Science
- 链接：https://doi.org/10.1016/0376-7388(95)00102-i
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 196. Networks of spiking neurons: The third generation of neural network models
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：3049
- 年份：1997
- 作者：Wolfgang Maass
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/s0893-6080(97)00011-7
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 197. Deep learning with coherent nanophotonic circuits
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：3045
- 年份：2017
- 作者：Yichen Shen, Nicholas C. Harris, Scott Skirlo et al.
- 来源：Nature Photonics
- 链接：https://doi.org/10.1038/nphoton.2017.93
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 198. Understanding of a convolutional neural network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：3044
- 年份：2017
- 作者：Saad Albawi, Tareq Abed Mohammed, Saad Al-Zawi
- 来源：2017 International Conference on Engineering and Technology (ICET)
- 链接：https://doi.org/10.1109/icengtechnol.2017.8308186
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 199. Deep Learning‐Based Crack Damage Detection Using Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：3038
- 年份：2017
- 作者：Young‐Jin Cha, Wooram Choi, Oral Büyüköztürk
- 来源：Computer-Aided Civil and Infrastructure Engineering
- 链接：https://doi.org/10.1111/mice.12263
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 200. Hierarchical Attention Networks for Document Classification
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：架构论文
- 引用数：3035
- 年份：2016
- 作者：Zichao Yang, Diyi Yang, Chris Dyer et al.
- 来源：Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
- 链接：https://doi.org/10.18653/v1/n16-1174
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 201. Long-term recurrent convolutional networks for visual recognition and description
- 类型：计算机视觉
- 标签：视觉, CNN, RNN
- 重要性：架构论文
- 引用数：3031
- 年份：2015
- 作者：Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama et al.
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7298878
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 202. Deep Face Recognition
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：3029
- 年份：2015
- 作者：Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman
- 来源：Procedings of the British Machine Vision Conference 2015
- 链接：https://doi.org/10.5244/c.29.41
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 203. T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
- 类型：计算机视觉 / 图学习 / 系统/框架
- 标签：视觉, Graph, Framework/System, CNN
- 重要性：系统/框架
- 引用数：3027
- 年份：2020
- 作者：Ling Zhao, Yujiao Song, Chao Zhang et al.
- 来源：IEEE Transactions on Intelligent Transportation Systems
- 链接：https://doi.org/10.1109/tits.2019.2935152
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 204. Learning to Prompt for Vision-Language Models
- 类型：NLP/语言模型 / 计算机视觉 / 多模态
- 标签：Language Model, 视觉, Multimodal
- 重要性：应用/方法论文
- 引用数：3025
- 年份：2022
- 作者：Kaiyang Zhou, Jingkang Yang, Chen Change Loy et al.
- 来源：International Journal of Computer Vision
- 链接：https://doi.org/10.1007/s11263-022-01653-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 205. Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：2998
- 年份：2017
- 作者：Konstantinos Kamnitsas, Christian Ledig, Virginia F.J. Newcombe et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2016.10.004
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 206. Probabilistic neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2986
- 年份：1990
- 作者：Donald F. Specht
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/0893-6080(90)90049-q
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 207. Wide &amp;amp; Deep Learning for Recommender Systems
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：2972
- 年份：2016
- 作者：Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen et al.
- 来源：Proceedings of the 1st Workshop on Deep Learning for Recommender Systems
- 链接：https://doi.org/10.1145/2988450.2988454
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 208. UNETR: Transformers for 3D Medical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：2942
- 年份：2022
- 作者：Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath et al.
- 来源：2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
- 链接：https://doi.org/10.1109/wacv51458.2022.00181
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 209. SuperPoint: Self-Supervised Interest Point Detection and Description
- 类型：计算机视觉
- 标签：视觉, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：2941
- 年份：2018
- 作者：Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- 链接：https://doi.org/10.1109/cvprw.2018.00060
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 210. Deep learning models for plant disease detection and diagnosis
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2934
- 年份：2018
- 作者：Konstantinos P. Ferentinos
- 来源：Computers and Electronics in Agriculture
- 链接：https://doi.org/10.1016/j.compag.2018.01.009
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 211. New Product Diffusion Models in Marketing: A Review and Directions for Research
- 类型：生成模型
- 标签：Generative AI, Diffusion/Flow
- 重要性：综述论文
- 引用数：2908
- 年份：1990
- 作者：Vijay Mahajan, Eitan Muller, Frank M. Bass
- 来源：Journal of Marketing
- 链接：https://doi.org/10.1177/002224299005400101
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 212. SignalP 6.0 predicts all five types of signal peptides using protein language models
- 类型：NLP/语言模型 / 医疗/生命科学AI
- 标签：Language Model, Medical AI
- 重要性：应用/方法论文
- 引用数：2889
- 年份：2022
- 作者：Felix Teufel, José Juan Almagro Armenteros, Alexander Rosenberg Johansen et al.
- 来源：Nature Biotechnology
- 链接：https://doi.org/10.1038/s41587-021-01156-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 213. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：2877
- 年份：2017
- 作者：Babak Ehteshami Bejnordi, Mitko Veta, Paul Johannes van Diest et al.
- 来源：JAMA
- 链接：https://doi.org/10.1001/jama.2017.14585
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 214. Road Extraction by Deep Residual U-Net
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection, Segmentation
- 重要性：奠基/方法论文
- 引用数：2866
- 年份：2018
- 作者：Zhengxin Zhang, Qingjie Liu, Yunhong Wang
- 来源：IEEE Geoscience and Remote Sensing Letters
- 链接：https://doi.org/10.1109/lgrs.2018.2802944
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 215. Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：2863
- 年份：2016
- 作者：Nima Tajbakhsh, Jae Y. Shin, Suryakanth R. Gurudu et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2535302
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 216. Deep Learning: Methods and Applications
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2862
- 年份：2014
- 作者：Li Deng, Dong Yu
- 来源：Foundations and Trends® in Signal Processing
- 链接：https://doi.org/10.1561/2000000039
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 217. Deep learning for time series classification: a review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2856
- 年份：2019
- 作者：Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber et al.
- 来源：Data Mining and Knowledge Discovery
- 链接：https://doi.org/10.1007/s10618-019-00619-1
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 218. Social LSTM: Human Trajectory Prediction in Crowded Spaces
- 类型：计算机视觉
- 标签：视觉, RNN
- 重要性：奠基/方法论文
- 引用数：2846
- 年份：2016
- 作者：Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.110
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 219. Predicting Splicing from Primary Sequence with Deep Learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2834
- 年份：2019
- 作者：Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou, Jeremy F. McRae et al.
- 来源：Cell
- 链接：https://doi.org/10.1016/j.cell.2018.12.015
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 220. A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：综述论文
- 引用数：2824
- 年份：2023
- 作者：Juan Terven, Diana-Margarita Córdova-Esparza, Julio-Alejandro Romero-González
- 来源：Machine Learning and Knowledge Extraction
- 链接：https://doi.org/10.3390/make5040083
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 221. Graph Convolutional Neural Networks for Web-Scale Recommender Systems
- 类型：计算机视觉 / 图学习 / 系统/框架
- 标签：视觉, Graph, Framework/System, CNN
- 重要性：系统/框架
- 引用数：2822
- 年份：2018
- 作者：Rex Ying, Ruining He, Kaifeng Chen et al.
- 来源：Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &amp;amp; Data Mining
- 链接：https://doi.org/10.1145/3219819.3219890
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 222. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2801
- 年份：2016
- 作者：Yushi Chen, Hanlu Jiang, Chunyang Li et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2016.2584107
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 223. Geometric Deep Learning: Going beyond Euclidean data
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2792
- 年份：2017
- 作者：Michael M. Bronstein, Joan Bruna, Yann LeCun et al.
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/msp.2017.2693418
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 224. Artificial neural networks (the multilayer perceptron)—a review of applications in the atmospheric sciences
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2762
- 年份：1998
- 作者：M.W Gardner, S.R Dorling
- 来源：Atmospheric Environment
- 链接：https://doi.org/10.1016/s1352-2310(97)00447-0
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 225. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2761
- 年份：2019
- 作者：Awni Y. Hannun, Pranav Rajpurkar, Masoumeh Haghpanahi et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-018-0268-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 226. A survey of deep neural network architectures and their applications
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2752
- 年份：2017
- 作者：Weibo Liu, Zidong Wang, Xiaohui Liu et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2016.12.038
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 227. Brain tumor segmentation with Deep Neural Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Unsupervised Representation, Segmentation
- 重要性：架构论文
- 引用数：2751
- 年份：2017
- 作者：Mohammad Havaei, Axel Davy, David Warde-Farley et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2016.05.004
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 228. CNN Features Off-the-Shelf: An Astounding Baseline for Recognition
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2748
- 年份：2014
- 作者：Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan et al.
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition Workshops
- 链接：https://doi.org/10.1109/cvprw.2014.131
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 229. Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：2743
- 年份：2015
- 作者：Babak Alipanahi, Andrew Delong, Matthew T Weirauch et al.
- 来源：Nature Biotechnology
- 链接：https://doi.org/10.1038/nbt.3300
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 230. A Survey on Evaluation of Large Language Models
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System
- 重要性：综述论文
- 引用数：2733
- 年份：2024
- 作者：Yupeng Chang, Xu Wang, Jindong Wang et al.
- 来源：ACM Transactions on Intelligent Systems and Technology
- 链接：https://doi.org/10.1145/3641289
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 231. Recent Trends in Deep Learning Based Natural Language Processing [Review Article]
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2721
- 年份：2018
- 作者：Tom Young, Devamanyu Hazarika, Soujanya Poria et al.
- 来源：IEEE Computational Intelligence Magazine
- 链接：https://doi.org/10.1109/mci.2018.2840738
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 232. Two-dimensional electron gases induced by spontaneous and piezoelectric polarization charges in N- and Ga-face AlGaN/GaN heterostructures
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：2720
- 年份：1999
- 作者：O. Ambacher, J. Smart, J. R. Shealy et al.
- 来源：Journal of Applied Physics
- 链接：https://doi.org/10.1063/1.369664
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 233. Are Transformers Effective for Time Series Forecasting?
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：2717
- 年份：2023
- 作者：Ailing Zeng, Muxi Chen, Lei Zhang et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v37i9.26317
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 234. Temporal Fusion Transformers for interpretable multi-horizon time series forecasting
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：2714
- 年份：2021
- 作者：Bryan Lim, Sercan Ö. Arık, Nicolas Loeff et al.
- 来源：International Journal of Forecasting
- 链接：https://doi.org/10.1016/j.ijforecast.2021.03.012
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 235. Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, Optimization/Training, CNN
- 重要性：架构论文
- 引用数：2712
- 年份：2018
- 作者：Aditya Chattopadhay, Anirban Sarkar, Prantik Howlader et al.
- 来源：2018 IEEE Winter Conference on Applications of Computer Vision (WACV)
- 链接：https://doi.org/10.1109/wacv.2018.00097
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 236. Artificial neural networks: fundamentals, computing, design, and application
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2703
- 年份：2000
- 作者：I.A Basheer, M Hajmeer
- 来源：Journal of Microbiological Methods
- 链接：https://doi.org/10.1016/s0167-7012(00)00201-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 237. Deep Neural Networks for YouTube Recommendations
- 类型：系统/框架
- 标签：Framework/System, Optimization/Training
- 重要性：系统/框架
- 引用数：2691
- 年份：2016
- 作者：Paul Covington, Jay Adams, Emre Sargin
- 来源：Proceedings of the 10th ACM Conference on Recommender Systems
- 链接：https://doi.org/10.1145/2959100.2959190
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 238. State-of-the-art in artificial neural network applications: A survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2684
- 年份：2018
- 作者：Oludare Isaac Abiodun, Aman Jantan, Abiodun Esther Omolara et al.
- 来源：Heliyon
- 链接：https://doi.org/10.1016/j.heliyon.2018.e00938
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 239. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units
- 类型：NLP/语言模型
- 标签：BERT, Transformer, Pretraining, Language Model
- 重要性：奠基/方法论文
- 引用数：2659
- 年份：2021
- 作者：Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai et al.
- 来源：IEEE/ACM Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/taslp.2021.3122291
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 240. Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：2651
- 年份：2018
- 作者：Nicolas Coudray, Paolo Santiago Ocampo, Theodore Sakellaropoulos et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-018-0177-5
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 241. Deep Learning for Computer Vision: A Brief Review
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：2644
- 年份：2018
- 作者：Athanasios Voulodimos, Nikolaos Doulamis, Anastasios Doulamis et al.
- 来源：Computational Intelligence and Neuroscience
- 链接：https://doi.org/10.1155/2018/7068349
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 242. Neural Networks and the Bias/Variance Dilemma
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2636
- 年份：1992
- 作者：Stuart Geman, Elie Bienenstock, René Doursat
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.1992.4.1.1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 243. VoxNet: A 3D Convolutional Neural Network for real-time object recognition
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：2629
- 年份：2015
- 作者：Daniel Maturana, Sebastian Scherer
- 来源：2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
- 链接：https://doi.org/10.1109/iros.2015.7353481
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 244. FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2625
- 年份：2018
- 作者：Kai Zhang, Wangmeng Zuo, Lei Zhang
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2018.2839891
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 245. An Introduction to Deep Learning for the Physical Layer
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2624
- 年份：2017
- 作者：Timothy O'Shea, Jakob Hoydis
- 来源：IEEE Transactions on Cognitive Communications and Networking
- 链接：https://doi.org/10.1109/tccn.2017.2758370
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 246. Deep learning and its applications to machine health monitoring
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：2622
- 年份：2019
- 作者：Rui Zhao, Ruqiang Yan, Zhenghua Chen et al.
- 来源：Mechanical Systems and Signal Processing
- 链接：https://doi.org/10.1016/j.ymssp.2018.05.050
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 247. Machine learning and deep learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2620
- 年份：2021
- 作者：Christian Janiesch, Patrick Zschech, Kai Heinrich
- 来源：Electronic Markets
- 链接：https://doi.org/10.1007/s12525-021-00475-2
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 248. Multi-view Convolutional Neural Networks for 3D Shape Recognition
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2613
- 年份：2015
- 作者：Hang Su, Subhransu Maji, Evangelos Kalogerakis et al.
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.114
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 249. A Review of Yolo Algorithm Developments
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：综述论文
- 引用数：2591
- 年份：2022
- 作者：Peiyuan Jiang, Daji Ergu, Fangyao Liu et al.
- 来源：Procedia Computer Science
- 链接：https://doi.org/10.1016/j.procs.2022.01.135
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 250. SuperGlue: Learning Feature Matching With Graph Neural Networks
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph
- 重要性：架构论文
- 引用数：2590
- 年份：2020
- 作者：Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.00499
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 251. Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2589
- 年份：2017
- 作者：Yu-Hsin Chen, Tushar Krishna, Joel S. Emer et al.
- 来源：IEEE Journal of Solid-State Circuits
- 链接：https://doi.org/10.1109/jssc.2016.2616357
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 252. The Limitations of Deep Learning in Adversarial Settings
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2589
- 年份：2016
- 作者：Nicolas Papernot, Patrick McDaniel, Somesh Jha et al.
- 来源：2016 IEEE European Symposium on Security and Privacy (EuroS&amp;amp;P)
- 链接：https://doi.org/10.1109/eurosp.2016.36
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 253. Survey on deep learning with class imbalance
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2588
- 年份：2019
- 作者：Justin M. Johnson, Taghi M. Khoshgoftaar
- 来源：Journal of Big Data
- 链接：https://doi.org/10.1186/s40537-019-0192-5
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 254. GaN, AlN, and InN: A review
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：综述论文
- 引用数：2586
- 年份：1992
- 作者：S. Strite, H. Morkoç
- 来源：Journal of Vacuum Science &amp;amp; Technology B: Microelectronics and Nanometer Structures Processing, Measurement, and Phenomena
- 链接：https://doi.org/10.1116/1.585897
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 255. An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2579
- 年份：2017
- 作者：Baoguang Shi, Xiang Bai, Cong Yao
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2016.2646371
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 256. 1D convolutional neural networks and applications: A survey
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：综述论文
- 引用数：2567
- 年份：2021
- 作者：Serkan Kiranyaz, Onur Avci, Osama Abdeljaber et al.
- 来源：Mechanical Systems and Signal Processing
- 链接：https://doi.org/10.1016/j.ymssp.2020.107398
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 257. Artificial neural networks: a tutorial
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2553
- 年份：1996
- 作者：A.K. Jain, Jianchang Mao, K.M. Mohiuddin
- 来源：Computer
- 链接：https://doi.org/10.1109/2.485891
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 258. Reservoir computing approaches to recurrent neural network training
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer, RNN
- 重要性：综述论文
- 引用数：2538
- 年份：2009
- 作者：Mantas Lukoševičius, Herbert Jaeger
- 来源：Computer Science Review
- 链接：https://doi.org/10.1016/j.cosrev.2009.03.005
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 259. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：2534
- 年份：2019
- 作者：Gabriele Campanella, Matthew G. Hanna, Luke Geneslaw et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-019-0508-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 260. Scalable and accurate deep learning with electronic health records
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2533
- 年份：2018
- 作者：Alvin Rajkomar, Eyal Oren, Kai Chen et al.
- 来源：npj Digital Medicine
- 链接：https://doi.org/10.1038/s41746-018-0029-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 261. Neural Machine Translation of Rare Words with Subword Units
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：2522
- 年份：2016
- 作者：Rico Sennrich, Barry Haddow, Alexandra Birch
- 来源：Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 链接：https://doi.org/10.18653/v1/p16-1162
- 概述：研究序列到序列/神经机器翻译模型，推动端到端NLP建模。

## 262. Scientific Machine Learning Through Physics–Informed Neural Networks: Where we are and What’s Next
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2519
- 年份：2022
- 作者：Salvatore Cuomo, Vincenzo Schiano Di Cola, Fabio Giampaolo et al.
- 来源：Journal of Scientific Computing
- 链接：https://doi.org/10.1007/s10915-022-01939-z
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 263. A systematic study of the class imbalance problem in convolutional neural networks
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN, Segmentation
- 重要性：系统/框架
- 引用数：2514
- 年份：2018
- 作者：Mateusz Buda, Atsuto Maki, Maciej A. Mazurowski
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2018.07.011
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 264. A survey of the recent architectures of deep convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：综述论文
- 引用数：2513
- 年份：2020
- 作者：Asifullah Khan, Anabia Sohail, Umme Zahoora et al.
- 来源：Artificial Intelligence Review
- 链接：https://doi.org/10.1007/s10462-020-09825-6
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 265. Region-Based Convolutional Networks for Accurate Object Detection and Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection, Segmentation
- 重要性：架构论文
- 引用数：2504
- 年份：2016
- 作者：Ross Girshick, Jeff Donahue, Trevor Darrell et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2015.2437384
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 266. Chain-Of-Thought Prompting Elicits Reasoning in Large Language Models
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System, Reasoning/Test-time Scaling
- 重要性：系统/框架
- 引用数：2496
- 年份：2022
- 作者：Jason Wei, Xuezhi Wang, Dale Schuurmans et al.
- 来源：Advances in Neural Information Processing Systems 35
- 链接：https://doi.org/10.52202/068431-1800
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 267. Heterogeneous Graph Attention Network
- 类型：图学习
- 标签：Graph, Attention/Transformer
- 重要性：架构论文
- 引用数：2487
- 年份：2019
- 作者：Xiao Wang, Houye Ji, Chuan Shi et al.
- 来源：The World Wide Web Conference
- 链接：https://doi.org/10.1145/3308558.3313562
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 268. Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：2480
- 年份：2019
- 作者：Shengnan Guo, Youfang Lin, Ning Feng et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v33i01.3301922
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 269. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：综述论文
- 引用数：2477
- 年份：2018
- 作者：Tian Xie, Jeffrey C. Grossman
- 来源：Physical Review Letters
- 链接：https://doi.org/10.1103/physrevlett.120.145301
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 270. Deep learning for healthcare: review, opportunities and challenges
- 类型：强化学习 / 医疗/生命科学AI
- 标签：RL, Medical AI
- 重要性：综述论文
- 引用数：2472
- 年份：2018
- 作者：Riccardo Miotto, Fei Wang, Shuang Wang et al.
- 来源：Briefings in Bioinformatics
- 链接：https://doi.org/10.1093/bib/bbx044
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 271. Point Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2464
- 年份：2021
- 作者：Hengshuang Zhao, Li Jiang, Jiaya Jia et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.01595
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 272. Deep Learning-Based Classification of Hyperspectral Data
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：2454
- 年份：2014
- 作者：Yushi Chen, Zhouhan Lin, Xing Zhao et al.
- 来源：IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- 链接：https://doi.org/10.1109/jstars.2014.2329330
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 273. A review of uncertainty quantification in deep learning: Techniques, applications and challenges
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2448
- 年份：2021
- 作者：Moloud Abdar, Farhad Pourpanah, Sadiq Hussain et al.
- 来源：Information Fusion
- 链接：https://doi.org/10.1016/j.inffus.2021.05.008
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 274. Continual lifelong learning with neural networks: A review
- 类型：深度学习相关
- 标签：Segmentation
- 重要性：综述论文
- 引用数：2432
- 年份：2019
- 作者：German I. Parisi, Ronald Kemker, Jose L. Part et al.
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2019.01.012
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 275. Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2413
- 年份：2015
- 作者：Wei-Long Zheng, Bao-Liang Lu
- 来源：IEEE Transactions on Autonomous Mental Development
- 链接：https://doi.org/10.1109/tamd.2015.2431497
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 276. Deep Learning for Anomaly Detection
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2408
- 年份：2022
- 作者：Guansong Pang, Chunhua Shen, Longbing Cao et al.
- 来源：ACM Computing Surveys
- 链接：https://doi.org/10.1145/3439950
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 277. Face recognition: a convolutional neural-network approach
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2402
- 年份：1997
- 作者：S. Lawrence, C.L. Giles, Ah Chung Tsoi et al.
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.554195
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 278. RepVGG: Making VGG-style ConvNets Great Again
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2400
- 年份：2021
- 作者：Xiaohan Ding, Xiangyu Zhang, Ningning Ma et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.01352
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 279. Evolving Neural Networks through Augmenting Topologies
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2397
- 年份：2002
- 作者：Kenneth O. Stanley, Risto Miikkulainen
- 来源：Evolutionary Computation
- 链接：https://doi.org/10.1162/106365602320169811
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 280. Brain Tumor Segmentation Using Convolutional Neural Networks in MRI Images
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：2394
- 年份：2016
- 作者：Sergio Pereira, Adriano Pinto, Victor Alves et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2538465
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 281. DeepPose: Human Pose Estimation via Deep Neural Networks
- 类型：计算机视觉 / 强化学习
- 标签：视觉, RL
- 重要性：架构论文
- 引用数：2391
- 年份：2014
- 作者：Alexander Toshev, Christian Szegedy
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2014.214
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 282. Deep Learning for Generic Object Detection: A Survey
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：综述论文
- 引用数：2386
- 年份：2020
- 作者：Li Liu, Wanli Ouyang, Xiaogang Wang et al.
- 来源：International Journal of Computer Vision
- 链接：https://doi.org/10.1007/s11263-019-01247-4
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 283. ViViT: A Video Vision Transformer
- 类型：计算机视觉 / 视频/世界模型
- 标签：视觉, Video, Attention/Transformer
- 重要性：架构论文
- 引用数：2386
- 年份：2021
- 作者：Anurag Arnab, Mostafa Dehghani, Georg Heigold et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00676
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 284. Deeply-Recursive Convolutional Network for Image Super-Resolution
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2381
- 年份：2016
- 作者：Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.181
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 285. Deep Learning: A Comprehensive Overview on Techniques, Taxonomy, Applications and Research Directions
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：2378
- 年份：2021
- 作者：Iqbal H. Sarker
- 来源：SN Computer Science
- 链接：https://doi.org/10.1007/s42979-021-00815-1
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 286. Recombinant Mouse OB Protein: Evidence for a Peripheral Signal Linking Adiposity and Central Neural Networks
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：2364
- 年份：1995
- 作者：L. Arthur Campfield, Françoise J. Smith, Yves Guisez et al.
- 来源：Science
- 链接：https://doi.org/10.1126/science.7624778
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 287. The Vanishing Gradient Problem During Learning Recurrent Neural Nets and Problem Solutions
- 类型：系统/框架
- 标签：Framework/System, Optimization/Training, RNN
- 重要性：系统/框架
- 引用数：2360
- 年份：1998
- 作者：Sepp Hochreiter
- 来源：International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems
- 链接：https://doi.org/10.1142/s0218488598000094
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 288. Neural network-based face detection
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2355
- 年份：1998
- 作者：H.A. Rowley, S. Baluja, T. Kanade
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/34.655647
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 289. All-optical machine learning using diffractive deep neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2352
- 年份：2018
- 作者：Xing Lin, Yair Rivenson, Nezih T. Yardimci et al.
- 来源：Science
- 链接：https://doi.org/10.1126/science.aat8084
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 290. Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition
- 类型：计算机视觉 / 多模态
- 标签：视觉, Multimodal, CNN, RNN
- 重要性：奠基/方法论文
- 引用数：2345
- 年份：2016
- 作者：Francisco Ordóñez, Daniel Roggen
- 来源：Sensors
- 链接：https://doi.org/10.3390/s16010115
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 291. Conformer: Convolution-augmented Transformer for Speech Recognition
- 类型：语音/音频
- 标签：Speech, Attention/Transformer
- 重要性：架构论文
- 引用数：2343
- 年份：2020
- 作者：Anmol Gulati, James Qin, Chung-Cheng Chiu et al.
- 来源：Interspeech 2020
- 链接：https://doi.org/10.21437/interspeech.2020-3015
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 292. Predicting effects of noncoding variants with deep learning–based sequence model
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2329
- 年份：2015
- 作者：Jian Zhou, Olga G Troyanskaya
- 来源：Nature Methods
- 链接：https://doi.org/10.1038/nmeth.3547
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 293. Loss Functions for Image Restoration With Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2328
- 年份：2017
- 作者：Hang Zhao, Orazio Gallo, Iuri Frosio et al.
- 来源：IEEE Transactions on Computational Imaging
- 链接：https://doi.org/10.1109/tci.2016.2644865
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 294. Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network
- 类型：深度学习相关
- 标签：RNN
- 重要性：奠基/方法论文
- 引用数：2324
- 年份：2019
- 作者：Weicong Kong, Zhao Yang Dong, Youwei Jia et al.
- 来源：IEEE Transactions on Smart Grid
- 链接：https://doi.org/10.1109/tsg.2017.2753802
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 295. Artificial neural networks for solving ordinary and partial differential equations
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2319
- 年份：1998
- 作者：I.E. Lagaris, A. Likas, D.I. Fotiadis
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.712178
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 296. Deep learning with long short-term memory networks for financial market predictions
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：2310
- 年份：2018
- 作者：Thomas Fischer, Christopher Krauss
- 来源：European Journal of Operational Research
- 链接：https://doi.org/10.1016/j.ejor.2017.11.054
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 297. Attention mechanisms in computer vision: A survey
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：综述论文
- 引用数：2308
- 年份：2022
- 作者：Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu et al.
- 来源：Computational Visual Media
- 链接：https://doi.org/10.1007/s41095-022-0271-y
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 298. MnasNet: Platform-Aware Neural Architecture Search for Mobile
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2285
- 年份：2019
- 作者：Mingxing Tan, Bo Chen, Ruoming Pang et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00293
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 299. CosFace: Large Margin Cosine Loss for Deep Face Recognition
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：2273
- 年份：2018
- 作者：Hao Wang, Yitong Wang, Zheng Zhou et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00552
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 300. Industry 4.0 and Industry 5.0—Inception, conception and perception
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：2270
- 年份：2021
- 作者：Xun Xu, Yuqian Lu, Birgit Vogel-Heuser et al.
- 来源：Journal of Manufacturing Systems
- 链接：https://doi.org/10.1016/j.jmsy.2021.10.006
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 301. An Introduction to Variational Autoencoders
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：2256
- 年份：2019
- 作者：Diederik P. Kingma, Max Welling
- 来源：Foundations and Trends® in Machine Learning
- 链接：https://doi.org/10.1561/2200000056
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 302. A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation
- 类型：计算机视觉
- 标签：视觉, CNN, Dataset/Benchmark
- 重要性：数据集/基准
- 引用数：2256
- 年份：2016
- 作者：Nikolaus Mayer, Eddy Ilg, Philip Hausser et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.438
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 303. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning
- 类型：深度学习相关
- 标签：Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：2249
- 年份：2022
- 作者：Ahmed Elnaggar, Michael Heinzinger, Christian Dallago et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2021.3095381
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 304. Deep learning in remote sensing applications: A meta-analysis and review
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：2245
- 年份：2019
- 作者：Lei Ma, Yu Liu, Xueliang Zhang et al.
- 来源：ISPRS Journal of Photogrammetry and Remote Sensing
- 链接：https://doi.org/10.1016/j.isprsjprs.2019.04.015
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 305. Swin Transformer V2: Scaling Up Capacity and Resolution
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2233
- 年份：2022
- 作者：Ze Liu, Han Hu, Yutong Lin et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01170
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 306. A Practical Bayesian Framework for Backpropagation Networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：2229
- 年份：1992
- 作者：David J. C. MacKay
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.1992.4.3.448
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 307. Domain randomization for transferring deep neural networks from simulation to the real world
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：2227
- 年份：2017
- 作者：Josh Tobin, Rachel Fong, Alex Ray et al.
- 来源：2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
- 链接：https://doi.org/10.1109/iros.2017.8202133
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 308. PVT v2: Improved baselines with pyramid vision transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2227
- 年份：2022
- 作者：Wenhai Wang, Enze Xie, Xiang Li et al.
- 来源：Computational Visual Media
- 链接：https://doi.org/10.1007/s41095-022-0274-8
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 309. Neural Architectures for Named Entity Recognition
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2225
- 年份：2016
- 作者：Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian et al.
- 来源：Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
- 链接：https://doi.org/10.18653/v1/n16-1030
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 310. Universal Language Model Fine-tuning for Text Classification
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：2225
- 年份：2018
- 作者：Jeremy Howard, Sebastian Ruder
- 来源：Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 链接：https://doi.org/10.18653/v1/p18-1031
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 311. VGGFace2: A Dataset for Recognising Faces across Pose and Age
- 类型：计算机视觉
- 标签：视觉, CNN, Dataset/Benchmark
- 重要性：数据集/基准
- 引用数：2224
- 年份：2018
- 作者：Qiong Cao, Li Shen, Weidi Xie et al.
- 来源：2018 13th IEEE International Conference on Automatic Face &amp;amp; Gesture Recognition (FG 2018)
- 链接：https://doi.org/10.1109/fg.2018.00020
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 312. Clinically applicable deep learning for diagnosis and referral in retinal disease
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：2222
- 年份：2018
- 作者：Jeffrey De Fauw, Joseph R. Ledsam, Bernardino Romera-Paredes et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-018-0107-6
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 313. Taming Transformers for High-Resolution Image Synthesis
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Attention/Transformer
- 重要性：架构论文
- 引用数：2204
- 年份：2021
- 作者：Patrick Esser, Robin Rombach, Bjorn Ommer
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.01268
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 314. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：2198
- 年份：2023
- 作者：Nataniel Ruiz, Yuanzhen Li, Varun Jampani et al.
- 来源：2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52729.2023.02155
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 315. DeepXDE: A Deep Learning Library for Solving Differential Equations
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：2190
- 年份：2021
- 作者：Lu Lu, Xuhui Meng, Zhiping Mao et al.
- 来源：SIAM Review
- 链接：https://doi.org/10.1137/19m1274067
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 316. Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：应用/方法论文
- 引用数：2183
- 年份：2018
- 作者：Olivier Bernard, Alain Lalande, Clement Zotti et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2018.2837502
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 317. FusionGAN: A generative adversarial network for infrared and visible image fusion
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：2173
- 年份：2019
- 作者：Jiayi Ma, Wei Yu, Pengwei Liang et al.
- 来源：Information Fusion
- 链接：https://doi.org/10.1016/j.inffus.2018.09.004
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 318. Pore and solid diffusion models for fixed‐bed adsorbers
- 类型：生成模型
- 标签：Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：2171
- 年份：1974
- 作者：Thomas W. Weber, Ranjit K. Chakravorti
- 来源：AIChE Journal
- 链接：https://doi.org/10.1002/aic.690200204
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 319. Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2170
- 年份：2023
- 作者：Jierun Chen, Shiu-hong Kao, Hao He et al.
- 来源：2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52729.2023.01157
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 320. Automated detection of COVID-19 cases using deep neural networks with X-ray images
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2169
- 年份：2020
- 作者：Tulin Ozturk, Muhammed Talo, Eylul Azra Yildirim et al.
- 来源：Computers in Biology and Medicine
- 链接：https://doi.org/10.1016/j.compbiomed.2020.103792
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 321. Energy and Policy Considerations for Deep Learning in NLP
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：2162
- 年份：2019
- 作者：Emma Strubell, Ananya Ganesh, Andrew McCallum
- 来源：Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics
- 链接：https://doi.org/10.18653/v1/p19-1355
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 322. Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images From Multiethnic Populations With Diabetes
- 类型：NLP/语言模型 / 计算机视觉 / 医疗/生命科学AI
- 标签：Language Model, 视觉, Medical AI, Framework/System, Attention/Transformer
- 重要性：系统/框架
- 引用数：2158
- 年份：2017
- 作者：Daniel Shu Wei Ting, Carol Yim-Lui Cheung, Gilbert Lim et al.
- 来源：JAMA
- 链接：https://doi.org/10.1001/jama.2017.18152
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 323. Image Segmentation Using Deep Learning: A Survey
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：综述论文
- 引用数：2150
- 年份：2021
- 作者：Shervin Minaee, Yuri Y. Boykov, Fatih Porikli et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2021.3059968
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 324. CvT: Introducing Convolutions to Vision Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2150
- 年份：2021
- 作者：Haiping Wu, Bin Xiao, Noel Codella et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00009
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 325. Understanding deep learning (still) requires rethinking generalization
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2149
- 年份：2021
- 作者：Chiyuan Zhang, Samy Bengio, Moritz Hardt et al.
- 来源：Communications of the ACM
- 链接：https://doi.org/10.1145/3446776
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 326. Deep learning for visual understanding: A review
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：2141
- 年份：2016
- 作者：Yanming Guo, Yu Liu, Ard Oerlemans et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2015.09.116
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 327. Context-Dependent Pre-Trained Deep Neural Networks for Large-Vocabulary Speech Recognition
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：2123
- 年份：2012
- 作者：G. E. Dahl, Dong Yu, Li Deng et al.
- 来源：IEEE Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/tasl.2011.2134090
- 概述：研究深度语音/音频建模，服务于识别、分离或表征学习任务。

## 328. Deep learning applications and challenges in big data analytics
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2119
- 年份：2015
- 作者：Maryam M Najafabadi, Flavio Villanustre, Taghi M Khoshgoftaar et al.
- 来源：Journal of Big Data
- 链接：https://doi.org/10.1186/s40537-014-0007-7
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 329. Deep Convolutional Neural Network for Inverse Problems in Imaging
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2106
- 年份：2017
- 作者：Kyong Hwan Jin, Michael T. McCann, Emmanuel Froustey et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2017.2713099
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 330. Barren plateaus in quantum neural network training landscapes
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：2106
- 年份：2018
- 作者：Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy et al.
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/s41467-018-07090-4
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 331. MultiResUNet : Rethinking the U-Net architecture for multimodal biomedical image segmentation
- 类型：计算机视觉 / 医疗/生命科学AI / 多模态
- 标签：视觉, Medical AI, Multimodal, Segmentation
- 重要性：架构论文
- 引用数：2103
- 年份：2020
- 作者：Nabil Ibtehaz, M. Sohel Rahman
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2019.08.025
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 332. Recurrent neural network based language model
- 类型：NLP/语言模型 / 语音/音频
- 标签：Language Model, Speech, RNN
- 重要性：架构论文
- 引用数：2097
- 年份：2010
- 作者：Tomáš Mikolov, Martin Karafiát, Lukáš Burget et al.
- 来源：Interspeech 2010
- 链接：https://doi.org/10.21437/interspeech.2010-343
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 333. Vision Transformers for Dense Prediction
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2094
- 年份：2021
- 作者：Rene Ranftl, Alexey Bochkovskiy, Vladlen Koltun
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.01196
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 334. Scalable Diffusion Models with Transformers
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Attention/Transformer, Diffusion/Flow
- 重要性：架构论文
- 引用数：2092
- 年份：2023
- 作者：William Peebles, Saining Xie
- 来源：2023 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv51070.2023.00387
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 335. Methods for interpreting and understanding deep neural networks
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：架构论文
- 引用数：2090
- 年份：2018
- 作者：Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller
- 来源：Digital Signal Processing
- 链接：https://doi.org/10.1016/j.dsp.2017.10.011
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 336. Robust deep learning–based protein sequence design using ProteinMPNN
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：2090
- 年份：2022
- 作者：J. Dauparas, I. Anishchenko, N. Bennett et al.
- 来源：Science
- 链接：https://doi.org/10.1126/science.add2187
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 337. GPT-3: Its Nature, Scope, Limits, and Consequences
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：架构论文
- 引用数：2090
- 年份：2020
- 作者：Luciano Floridi, Massimo Chiriatti
- 来源：Minds and Machines
- 链接：https://doi.org/10.1007/s11023-020-09548-1
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 338. A Learning Algorithm for Boltzmann Machines*
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：2089
- 年份：1985
- 作者：David H. Ackley, Geoffrey E. Hinton, Terrence J. Sejnowski
- 来源：Cognitive Science
- 链接：https://doi.org/10.1207/s15516709cog0901_7
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 339. A Deep Learning Approach to Antibiotic Discovery
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2087
- 年份：2020
- 作者：Jonathan M. Stokes, Kevin Yang, Kyle Swanson et al.
- 来源：Cell
- 链接：https://doi.org/10.1016/j.cell.2020.01.021
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 340. Multi-column deep neural networks for image classification
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：2086
- 年份：2012
- 作者：D. Ciresan, U. Meier, J. Schmidhuber
- 来源：2012 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2012.6248110
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 341. AlGaN/GaN HEMTs-an overview of device operation and applications
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：综述论文
- 引用数：2085
- 年份：2002
- 作者：U.K. Mishra, P. Parikh, Yi-Feng Wu
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/jproc.2002.1021567
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 342. Selective Attention Gates Visual Processing in the Extrastriate Cortex
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2073
- 年份：1985
- 作者：Jeffrey Moran, Robert Desimone
- 来源：Science
- 链接：https://doi.org/10.1126/science.4023713
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 343. Learning Efficient Convolutional Networks through Network Slimming
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2064
- 年份：2017
- 作者：Zhuang Liu, Jianguo Li, Zhiqiang Shen et al.
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.298
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 344. Learning and Transferring Mid-level Image Representations Using Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2062
- 年份：2014
- 作者：Maxime Oquab, Leon Bottou, Ivan Laptev et al.
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2014.222
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 345. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2062
- 年份：2016
- 作者：Relja Arandjelovic, Petr Gronat, Akihiko Torii et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.572
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 346. Digging Into Self-Supervised Monocular Depth Estimation
- 类型：计算机视觉
- 标签：视觉, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：2061
- 年份：2019
- 作者：Clement Godard, Oisin Mac Aodha, Michael Firman et al.
- 来源：2019 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2019.00393
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 347. A New Convolutional Neural Network-Based Data-Driven Fault Diagnosis Method
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2056
- 年份：2018
- 作者：Long Wen, Xinyu Li, Liang Gao et al.
- 来源：IEEE Transactions on Industrial Electronics
- 链接：https://doi.org/10.1109/tie.2017.2774777
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 348. Frustum PointNets for 3D Object Detection from RGB-D Data
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：应用/方法论文
- 引用数：2052
- 年份：2018
- 作者：Charles R. Qi, Wei Liu, Chenxia Wu et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00102
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 349. Uformer: A General U-Shaped Transformer for Image Restoration
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：2038
- 年份：2022
- 作者：Zhendong Wang, Xiaodong Cun, Jianmin Bao et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01716
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 350. DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：2033
- 年份：2018
- 作者：Han Wang, Linfeng Zhang, Jiequn Han et al.
- 来源：Computer Physics Communications
- 链接：https://doi.org/10.1016/j.cpc.2018.03.016
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 351. Fully hardware-implemented memristor convolutional neural network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2032
- 年份：2020
- 作者：Peng Yao, Huaqiang Wu, Bin Gao et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-020-1942-4
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 352. StackGAN: Text to Photo-Realistic Image Synthesis with Stacked Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：2024
- 年份：2017
- 作者：Han Zhang, Tao Xu, Hongsheng Li et al.
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.629
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 353. Learning Multi-domain Convolutional Neural Networks for Visual Tracking
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：2022
- 年份：2016
- 作者：Hyeonseob Nam, Bohyung Han
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.465
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 354. Deep Learning for Remote Sensing Data: A Technical Tutorial on the State of the Art
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：2021
- 年份：2016
- 作者：Liangpei Zhang, Lefei Zhang, Bo Du
- 来源：IEEE Geoscience and Remote Sensing Magazine
- 链接：https://doi.org/10.1109/mgrs.2016.2540798
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 355. Long short-term memory neural network for traffic speed prediction using remote microwave sensor data
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：2010
- 年份：2015
- 作者：Xiaolei Ma, Zhimin Tao, Yinhai Wang et al.
- 来源：Transportation Research Part C: Emerging Technologies
- 链接：https://doi.org/10.1016/j.trc.2015.03.014
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 356. Cellular neural networks: applications
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：2002
- 年份：1988
- 作者：L.O. Chua, L. Yang
- 来源：IEEE Transactions on Circuits and Systems
- 链接：https://doi.org/10.1109/31.7601
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 357. ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, Segmentation
- 重要性：系统/框架
- 引用数：2001
- 年份：2020
- 作者：Foivos I. Diakogiannis, François Waldner, Peter Caccetta et al.
- 来源：ISPRS Journal of Photogrammetry and Remote Sensing
- 链接：https://doi.org/10.1016/j.isprsjprs.2020.01.013
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 358. A Convolutional Neural Network for Modelling Sentences
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, CNN
- 重要性：架构论文
- 引用数：1987
- 年份：2014
- 作者：Nal Kalchbrenner, Edward Grefenstette, Phil Blunsom
- 来源：Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 链接：https://doi.org/10.3115/v1/p14-1062
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 359. Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1985
- 年份：2016
- 作者：Nicolas Papernot, Patrick McDaniel, Xi Wu et al.
- 来源：2016 IEEE Symposium on Security and Privacy (SP)
- 链接：https://doi.org/10.1109/sp.2016.41
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 360. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：1982
- 年份：2018
- 作者：Agrim Gupta, Justin Johnson, Li Fei-Fei et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00240
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 361. Deep Learning for 3D Point Clouds: A Survey
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：1979
- 年份：2021
- 作者：Yulan Guo, Hanyun Wang, Qingyong Hu et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2020.3005434
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 362. PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：1974
- 年份：2015
- 作者：Alex Kendall, Matthew Grimes, Roberto Cipolla
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.336
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 363. SciBERT: A Pretrained Language Model for Scientific Text
- 类型：NLP/语言模型
- 标签：BERT, Transformer, Pretraining, Language Model
- 重要性：奠基/方法论文
- 引用数：1972
- 年份：2019
- 作者：Iz Beltagy, Kyle Lo, Arman Cohan
- 来源：Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)
- 链接：https://doi.org/10.18653/v1/d19-1371
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 364. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System
- 重要性：综述论文
- 引用数：1969
- 年份：2025
- 作者：Lei Huang, Weijiang Yu, Weitao Ma et al.
- 来源：ACM Transactions on Information Systems
- 链接：https://doi.org/10.1145/3703155
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 365. Repetition and the brain: neural models of stimulus-specific effects
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1967
- 年份：2006
- 作者：Kalanit Grill-Spector, Richard Henson, Alex Martin
- 来源：Trends in Cognitive Sciences
- 链接：https://doi.org/10.1016/j.tics.2005.11.006
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 366. Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Dataset/Benchmark
- 重要性：架构论文
- 引用数：1965
- 年份：2021
- 作者：Li Yuan, Yunpeng Chen, Tao Wang et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00060
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 367. Evolving artificial neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1962
- 年份：1999
- 作者：Xin Yao
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/5.784219
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 368. Solving the quantum many-body problem with artificial neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1962
- 年份：2017
- 作者：Giuseppe Carleo, Matthias Troyer
- 来源：Science
- 链接：https://doi.org/10.1126/science.aag2302
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 369. TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Object Detection
- 重要性：架构论文
- 引用数：1959
- 年份：2021
- 作者：Xingkui Zhu, Shuchang Lyu, Xu Wang et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)
- 链接：https://doi.org/10.1109/iccvw54120.2021.00312
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 370. A direct adaptive method for faster backpropagation learning: the RPROP algorithm
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1956
- 年份：2002
- 作者：M. Riedmiller, H. Braun
- 来源：IEEE International Conference on Neural Networks
- 链接：https://doi.org/10.1109/icnn.1993.298623
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 371. Leptin activates anorexigenic POMC neurons through a neural network in the arcuate nucleus
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1953
- 年份：2001
- 作者：Michael A. Cowley, James L. Smart, Marcelo Rubinstein et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/35078085
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 372. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1950
- 年份：2014
- 作者：Wei Li, Rui Zhao, Tong Xiao et al.
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2014.27
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 373. Tabular data: Deep learning is not all you need
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1947
- 年份：2022
- 作者：Ravid Shwartz-Ziv, Amitai Armon
- 来源：Information Fusion
- 链接：https://doi.org/10.1016/j.inffus.2021.11.011
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 374. How Does ChatGPT Perform on the United States Medical Licensing Examination (USMLE)? The Implications of Large Language Models for Medical Education and Knowledge Assessment
- 类型：NLP/语言模型 / 医疗/生命科学AI
- 标签：Language Model, Medical AI, Attention/Transformer
- 重要性：架构论文
- 引用数：1945
- 年份：2023
- 作者：Aidan Gilson, Conrad W Safranek, Thomas Huang et al.
- 来源：JMIR Medical Education
- 链接：https://doi.org/10.2196/45312
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 375. SchNet – A deep learning architecture for molecules and materials
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1938
- 年份：2018
- 作者：K. T. Schütt, H. E. Sauceda, P.-J. Kindermans et al.
- 来源：The Journal of Chemical Physics
- 链接：https://doi.org/10.1063/1.5019779
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 376. Exceptional points enhance sensing in an optical microcavity
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：应用/方法论文
- 引用数：1936
- 年份：2017
- 作者：Weijian Chen, Şahin Kaya Özdemir, Guangming Zhao et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/nature23281
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 377. Opportunities and obstacles for deep learning in biology and medicine
- 类型：计算机视觉 / 强化学习
- 标签：视觉, RL
- 重要性：应用/方法论文
- 引用数：1929
- 年份：2018
- 作者：Travers Ching, Daniel S. Himmelstein, Brett K. Beaulieu-Jones et al.
- 来源：Journal of The Royal Society Interface
- 链接：https://doi.org/10.1098/rsif.2017.0387
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 378. Metalorganic vapor phase epitaxial growth of a high quality GaN film using an AlN buffer layer
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1927
- 年份：1986
- 作者：H. Amano, N. Sawaki, I. Akasaki et al.
- 来源：Applied Physics Letters
- 链接：https://doi.org/10.1063/1.96549
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 379. PCT: Point cloud transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1924
- 年份：2021
- 作者：Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu et al.
- 来源：Computational Visual Media
- 链接：https://doi.org/10.1007/s41095-021-0229-5
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 380. Deep Reinforcement Learning for Autonomous Driving: A Survey
- 类型：强化学习 / 系统/框架
- 标签：RL, Framework/System
- 重要性：综述论文
- 引用数：1923
- 年份：2022
- 作者：B Ravi Kiran, Ibrahim Sobh, Victor Talpaert et al.
- 来源：IEEE Transactions on Intelligent Transportation Systems
- 链接：https://doi.org/10.1109/tits.2021.3054625
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 381. Ensemble deep learning: A review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1917
- 年份：2022
- 作者：M.A. Ganaie, Minghui Hu, A.K. Malik et al.
- 来源：Engineering Applications of Artificial Intelligence
- 链接：https://doi.org/10.1016/j.engappai.2022.105151
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 382. Segmenter: Transformer for Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：1909
- 年份：2021
- 作者：Robin Strudel, Ricardo Garcia, Ivan Laptev et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00717
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 383. Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1902
- 年份：2017
- 作者：Seungjun Nah, Tae Hyun Kim, Kyoung Mu Lee
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.35
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 384. CNN architectures for large-scale audio classification
- 类型：计算机视觉 / 语音/音频
- 标签：视觉, Speech, CNN
- 重要性：架构论文
- 引用数：1901
- 年份：2017
- 作者：Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis et al.
- 来源：2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- 链接：https://doi.org/10.1109/icassp.2017.7952132
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 385. Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1899
- 年份：2019
- 作者：Tristan Bepler, Andrew Morin, Micah Rapp et al.
- 来源：Nature Methods
- 链接：https://doi.org/10.1038/s41592-019-0575-8
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 386. Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1899
- 年份：2019
- 作者：Shunping Ji, Shiqing Wei, Meng Lu
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2018.2858817
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 387. Absolute stability of global pattern formation and parallel memory storage by competitive neural networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1897
- 年份：1983
- 作者：Michael A. Cohen, Stephen Grossberg
- 来源：IEEE Transactions on Systems, Man, and Cybernetics
- 链接：https://doi.org/10.1109/tsmc.1983.6313075
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 388. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：架构论文
- 引用数：1897
- 年份：2019
- 作者：Zihang Dai, Zhilin Yang, Yiming Yang et al.
- 来源：Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics
- 链接：https://doi.org/10.18653/v1/p19-1285
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 389. Gate-variants of Gated Recurrent Unit (GRU) neural networks
- 类型：系统/框架
- 标签：Framework/System, RNN
- 重要性：系统/框架
- 引用数：1892
- 年份：2017
- 作者：Rahul Dey, Fathi M. Salem
- 来源：2017 IEEE 60th International Midwest Symposium on Circuits and Systems (MWSCAS)
- 链接：https://doi.org/10.1109/mwscas.2017.8053243
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 390. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1875
- 年份：2022
- 作者：Jonathan T. Barron, Ben Mildenhall, Dor Verbin et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00539
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 391. Channel Pruning for Accelerating Very Deep Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1870
- 年份：2017
- 作者：Yihui He, Xiangyu Zhang, Jian Sun
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.155
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 392. Cyclical Learning Rates for Training Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1870
- 年份：2017
- 作者：Leslie N. Smith
- 来源：2017 IEEE Winter Conference on Applications of Computer Vision (WACV)
- 链接：https://doi.org/10.1109/wacv.2017.58
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 393. Spectral–Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, Residual/Skip Connection
- 重要性：系统/框架
- 引用数：1867
- 年份：2018
- 作者：Zilong Zhong, Jonathan Li, Zhiming Luo et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2017.2755542
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 394. An overview of deep learning in medical imaging focusing on MRI
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：综述论文
- 引用数：1867
- 年份：2019
- 作者：Alexander Selvikvåg Lundervold, Arvid Lundervold
- 来源：Zeitschrift für Medizinische Physik
- 链接：https://doi.org/10.1016/j.zemedi.2018.11.002
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 395. Single-Image Crowd Counting via Multi-Column Convolutional Neural Network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1862
- 年份：2016
- 作者：Yingying Zhang, Desen Zhou, Siqin Chen et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.70
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 396. Deep Learning for Hyperspectral Image Classification: An Overview
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：1859
- 年份：2019
- 作者：Shutao Li, Weiwei Song, Leyuan Fang et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2019.2907932
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 397. DGM: A deep learning algorithm for solving partial differential equations
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1855
- 年份：2018
- 作者：Justin Sirignano, Konstantinos Spiliopoulos
- 来源：Journal of Computational Physics
- 链接：https://doi.org/10.1016/j.jcp.2018.08.029
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 398. Video Swin Transformer
- 类型：计算机视觉 / 视频/世界模型
- 标签：视觉, Video, Attention/Transformer
- 重要性：架构论文
- 引用数：1855
- 年份：2022
- 作者：Ze Liu, Jia Ning, Yue Cao et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00320
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 399. Physics-informed neural networks (PINNs) for fluid mechanics: a review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1853
- 年份：2021
- 作者：Shengze Cai, Zhiping Mao, Zhicheng Wang et al.
- 来源：Acta Mechanica Sinica
- 链接：https://doi.org/10.1007/s10409-021-01148-1
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 400. COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1852
- 年份：2020
- 作者：Linda Wang, Zhong Qiu Lin, Alexander Wong
- 来源：Scientific Reports
- 链接：https://doi.org/10.1038/s41598-020-76550-z
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 401. Neural networks for the prediction and forecasting of water resources variables: a review of modelling issues and applications
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1850
- 年份：2000
- 作者：Holger R. Maier, Graeme C. Dandy
- 来源：Environmental Modelling &amp;amp; Software
- 链接：https://doi.org/10.1016/s1364-8152(99)00007-9
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 402. Pre-Trained Image Processing Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1849
- 年份：2021
- 作者：Hanting Chen, Yunhe Wang, Tianyu Guo et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.01212
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 403. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1848
- 年份：2021
- 作者：Chun-Fu Richard Chen, Quanfu Fan, Rameswar Panda
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00041
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 404. Convolutional Neural Networks for Speech Recognition
- 类型：计算机视觉 / 语音/音频
- 标签：视觉, Speech, CNN
- 重要性：架构论文
- 引用数：1846
- 年份：2014
- 作者：Ossama Abdel-Hamid, Abdel-rahman Mohamed, Hui Jiang et al.
- 来源：IEEE/ACM Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/taslp.2014.2339736
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 405. Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks
- 类型：深度学习相关
- 标签：Optimization/Training
- 重要性：架构论文
- 引用数：1845
- 年份：2021
- 作者：Sifan Wang, Yujun Teng, Paris Perdikaris
- 来源：SIAM Journal on Scientific Computing
- 链接：https://doi.org/10.1137/20m1318043
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 406. Applications of Deep Reinforcement Learning in Communications and Networking: A Survey
- 类型：强化学习
- 标签：RL
- 重要性：综述论文
- 引用数：1840
- 年份：2019
- 作者：Nguyen Cong Luong, Dinh Thai Hoang, Shimin Gong et al.
- 来源：IEEE Communications Surveys &amp;amp; Tutorials
- 链接：https://doi.org/10.1109/comst.2019.2916583
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 407. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1833
- 年份：2015
- 作者：Anh Nguyen, Jason Yosinski, Jeff Clune
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7298640
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 408. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning
- 类型：强化学习
- 标签：RL, Segmentation
- 重要性：架构论文
- 引用数：1832
- 年份：2018
- 作者：Stefan Elfwing, Eiji Uchibe, Kenji Doya
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2017.12.012
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 409. Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1830
- 年份：2018
- 作者：C. L. Philip Chen, Zhulin Liu
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2017.2716952
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 410. Luminescence properties of defects in GaN
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1830
- 年份：2005
- 作者：Michael A. Reshchikov, Hadis Morkoç
- 来源：Journal of Applied Physics
- 链接：https://doi.org/10.1063/1.1868059
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 411. Attention gated networks: Learning to leverage salient regions in medical images
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer
- 重要性：架构论文
- 引用数：1829
- 年份：2019
- 作者：Jo Schlemper, Ozan Oktay, Michiel Schaap et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2019.01.012
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 412. SimCSE: Simple Contrastive Learning of Sentence Embeddings
- 类型：NLP/语言模型
- 标签：Language Model, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：1824
- 年份：2021
- 作者：Tianyu Gao, Xingcheng Yao, Danqi Chen
- 来源：Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing
- 链接：https://doi.org/10.18653/v1/2021.emnlp-main.552
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 413. End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：1824
- 年份：2019
- 作者：Diego Ardila, Atilla P. Kiraly, Sujeeth Bharadwaj et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-019-0447-x
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 414. Deep Learning for Person Re-Identification: A Survey and Outlook
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1820
- 年份：2022
- 作者：Mang Ye, Jianbing Shen, Gaojie Lin et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2021.3054775
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 415. LLNet: A deep autoencoder approach to natural low-light image enhancement
- 类型：计算机视觉
- 标签：视觉, Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：1809
- 年份：2017
- 作者：Kin Gwn Lore, Adedotun Akintayo, Soumik Sarkar
- 来源：Pattern Recognition
- 链接：https://doi.org/10.1016/j.patcog.2016.06.008
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 416. Covid-19: automatic detection from X-ray images utilizing transfer learning with convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1806
- 年份：2020
- 作者：Ioannis D. Apostolopoulos, Tzani A. Mpesiana
- 来源：Physical and Engineering Sciences in Medicine
- 链接：https://doi.org/10.1007/s13246-020-00865-4
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 417. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1806
- 年份：2021
- 作者：Jonathan T. Barron, Ben Mildenhall, Matthew Tancik et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00580
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 418. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1802
- 年份：2017
- 作者：Huifeng Guo, Ruiming TANG, Yunming Ye et al.
- 来源：Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence
- 链接：https://doi.org/10.24963/ijcai.2017/239
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 419. U-Net and Its Variants for Medical Image Segmentation: A Review of Theory and Applications
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：综述论文
- 引用数：1801
- 年份：2021
- 作者：Nahian Siddique, Sidike Paheding, Colin P. Elkin et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2021.3086020
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 420. 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1799
- 年份：2019
- 作者：Christopher Choy, JunYoung Gwak, Silvio Savarese
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00319
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 421. Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1796
- 年份：2020
- 作者：Zonghan Wu, Shirui Pan, Guodong Long et al.
- 来源：Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp;amp; Data Mining
- 链接：https://doi.org/10.1145/3394486.3403118
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 422. U-Net: deep learning for cell counting, detection, and morphometry
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：1792
- 年份：2019
- 作者：Thorsten Falk, Dominic Mai, Robert Bensch et al.
- 来源：Nature Methods
- 链接：https://doi.org/10.1038/s41592-018-0261-2
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 423. HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1790
- 年份：2020
- 作者：Swalpa Kumar Roy, Gopal Krishna, Shiv Ram Dubey et al.
- 来源：IEEE Geoscience and Remote Sensing Letters
- 链接：https://doi.org/10.1109/lgrs.2019.2918719
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 424. Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1787
- 年份：2018
- 作者：Hao Ye, Geoffrey Ye Li, Biing-Hwang Juang
- 来源：IEEE Wireless Communications Letters
- 链接：https://doi.org/10.1109/lwc.2017.2757490
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 425. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1784
- 年份：2018
- 作者：Guokun Lai, Wei-Cheng Chang, Yiming Yang et al.
- 来源：The 41st International ACM SIGIR Conference on Research &amp;amp; Development in Information Retrieval
- 链接：https://doi.org/10.1145/3209978.3210006
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 426. Conditional Random Fields as Recurrent Neural Networks
- 类型：计算机视觉
- 标签：视觉, RNN
- 重要性：架构论文
- 引用数：1778
- 年份：2015
- 作者：Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes et al.
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.179
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 427. Logistic regression and artificial neural network classification models: a methodology review
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：综述论文
- 引用数：1770
- 年份：2002
- 作者：Stephan Dreiseitl, Lucila Ohno-Machado
- 来源：Journal of Biomedical Informatics
- 链接：https://doi.org/10.1016/s1532-0464(03)00034-0
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 428. Neural networks for short-term load forecasting: a review and evaluation
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1770
- 年份：2001
- 作者：H.S. Hippert, C.E. Pedreira, R.C. Souza
- 来源：IEEE Transactions on Power Systems
- 链接：https://doi.org/10.1109/59.910780
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 429. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
- 类型：语音/音频
- 标签：Speech, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：1770
- 年份：2022
- 作者：Sanyuan Chen, Chengyi Wang, Zhengyang Chen et al.
- 来源：IEEE Journal of Selected Topics in Signal Processing
- 链接：https://doi.org/10.1109/jstsp.2022.3188113
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 430. Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition
- 类型：计算机视觉 / 视频/世界模型 / 图学习
- 标签：视觉, Video, Graph, CNN
- 重要性：架构论文
- 引用数：1769
- 年份：2019
- 作者：Lei Shi, Yifan Zhang, Jian Cheng et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.01230
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 431. Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：架构论文
- 引用数：1765
- 年份：2018
- 作者：Qimai Li, Zhichao Han, Xiao-ming Wu
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v32i1.11604
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 432. Diffusion Models in Vision: A Survey
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：综述论文
- 引用数：1765
- 年份：2023
- 作者：Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2023.3261988
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 433. A Deep Learning Approach for Intrusion Detection Using Recurrent Neural Networks
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1763
- 年份：2017
- 作者：Chuanlong Yin, Yuefei Zhu, Jinlong Fei et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2017.2762418
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 434. EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification
- 类型：计算机视觉
- 标签：视觉, Dataset/Benchmark
- 重要性：数据集/基准
- 引用数：1754
- 年份：2019
- 作者：Patrick Helber, Benjamin Bischke, Andreas Dengel et al.
- 来源：IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- 链接：https://doi.org/10.1109/jstars.2019.2918242
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 435. Recurrent Neural Networks for Multivariate Time Series with Missing Values
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1754
- 年份：2018
- 作者：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho et al.
- 来源：Scientific Reports
- 链接：https://doi.org/10.1038/s41598-018-24271-9
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 436. A universal SNP and small-indel variant caller using deep neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1753
- 年份：2018
- 作者：Ryan Poplin, Pi-Chuan Chang, David Alexander et al.
- 来源：Nature Biotechnology
- 链接：https://doi.org/10.1038/nbt.4235
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 437. ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1734
- 年份：2017
- 作者：J. S. Smith, O. Isayev, A. E. Roitberg
- 来源：Chemical Science
- 链接：https://doi.org/10.1039/c6sc05720a
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 438. Temporal Convolutional Networks for Action Segmentation and Detection
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：1730
- 年份：2017
- 作者：Colin Lea, Michael D. Flynn, Rene Vidal et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.113
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 439. Revisiting Unreasonable Effectiveness of Data in Deep Learning Era
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1727
- 年份：2017
- 作者：Chen Sun, Abhinav Shrivastava, Saurabh Singh et al.
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.97
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 440. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials
- 类型：NLP/语言模型 / 图学习
- 标签：Language Model, Graph, Attention/Transformer
- 重要性：架构论文
- 引用数：1719
- 年份：2022
- 作者：Simon Batzner, Albert Musaelian, Lixin Sun et al.
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/s41467-022-29939-5
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 441. 3D ShapeNets: A deep representation for volumetric shapes
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1713
- 年份：2015
- 作者：Zhirong Wu, Shuran Song, Aditya Khosla et al.
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7298801
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 442. Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：1713
- 年份：2016
- 作者：Serkan Kiranyaz, Turker Ince, Moncef Gabbouj
- 来源：IEEE Transactions on Biomedical Engineering
- 链接：https://doi.org/10.1109/tbme.2015.2468589
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 443. Graph convolutional networks: a comprehensive review
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：综述论文
- 引用数：1712
- 年份：2019
- 作者：Si Zhang, Hanghang Tong, Jiejun Xu et al.
- 来源：Computational Social Networks
- 链接：https://doi.org/10.1186/s40649-019-0069-y
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 444. Deep Convolutional Neural Networks for Hyperspectral Image Classification
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1709
- 年份：2015
- 作者：Wei Hu, Yangyu Huang, Li Wei et al.
- 来源：Journal of Sensors
- 链接：https://doi.org/10.1155/2015/258619
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 445. One Pixel Attack for Fooling Deep Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1706
- 年份：2019
- 作者：Jiawei Su, Danilo Vasconcellos Vargas, Kouichi Sakurai
- 来源：IEEE Transactions on Evolutionary Computation
- 链接：https://doi.org/10.1109/tevc.2019.2890858
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 446. Shortcut learning in deep neural networks
- 类型：NLP/语言模型 / 语音/音频
- 标签：Language Model, Speech, Attention/Transformer, Residual/Skip Connection
- 重要性：架构论文
- 引用数：1704
- 年份：2020
- 作者：Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis et al.
- 来源：Nature Machine Intelligence
- 链接：https://doi.org/10.1038/s42256-020-00257-z
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 447. Review on Convolutional Neural Networks (CNN) in vegetation remote sensing
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：综述论文
- 引用数：1704
- 年份：2021
- 作者：Teja Kattenborn, Jens Leitloff, Felix Schiefer et al.
- 来源：ISPRS Journal of Photogrammetry and Remote Sensing
- 链接：https://doi.org/10.1016/j.isprsjprs.2020.12.010
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 448. GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification
- 类型：计算机视觉 / 生成模型 / 医疗/生命科学AI
- 标签：视觉, Generative AI, Medical AI, CNN, GAN
- 重要性：架构论文
- 引用数：1703
- 年份：2018
- 作者：Maayan Frid-Adar, Idit Diamant, Eyal Klang et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2018.09.013
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 449. Bilinear CNN Models for Fine-Grained Visual Recognition
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1702
- 年份：2015
- 作者：Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.170
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 450. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network
- 类型：医疗/生命科学AI / 系统/框架
- 标签：Medical AI, Framework/System
- 重要性：系统/框架
- 引用数：1701
- 年份：2018
- 作者：Jared L. Katzman, Uri Shaham, Alexander Cloninger et al.
- 来源：BMC Medical Research Methodology
- 链接：https://doi.org/10.1186/s12874-018-0482-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 451. P-Type Conduction in Mg-Doped GaN Treated with Low-Energy Electron Beam Irradiation (LEEBI)
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1699
- 年份：1989
- 作者：Hiroshi Amano, Masahiro Kito, Kazumasa Hiramatsu et al.
- 来源：Japanese Journal of Applied Physics
- 链接：https://doi.org/10.1143/jjap.28.l2112
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 452. Phoneme recognition using time-delay neural networks
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：1695
- 年份：1989
- 作者：A. Waibel, T. Hanazawa, G. Hinton et al.
- 来源：IEEE Transactions on Acoustics, Speech, and Signal Processing
- 链接：https://doi.org/10.1109/29.21701
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 453. Meta-analytic evidence for common and distinct neural networks associated with directly experienced pain and empathy for pain
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1686
- 年份：2011
- 作者：Claus Lamm, Jean Decety, Tania Singer
- 来源：NeuroImage
- 链接：https://doi.org/10.1016/j.neuroimage.2010.10.014
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 454. Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：1686
- 年份：2018
- 作者：Frederik Kratzert, Daniel Klotz, Claire Brenner et al.
- 来源：Hydrology and Earth System Sciences
- 链接：https://doi.org/10.5194/hess-22-6005-2018
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 455. Learning Deep CNN Denoiser Prior for Image Restoration
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1678
- 年份：2017
- 作者：Kai Zhang, Wangmeng Zuo, Shuhang Gu et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.300
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 456. Conditional Prompt Learning for Vision-Language Models
- 类型：NLP/语言模型 / 计算机视觉 / 多模态
- 标签：Language Model, 视觉, Multimodal
- 重要性：应用/方法论文
- 引用数：1678
- 年份：2022
- 作者：Kaiyang Zhou, Jingkang Yang, Chen Change Loy et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01631
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 457. Basic concepts of artificial neural network (ANN) modeling and its application in pharmaceutical research
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1678
- 年份：2000
- 作者：S Agatonovic-Kustrin, R Beresford
- 来源：Journal of Pharmaceutical and Biomedical Analysis
- 链接：https://doi.org/10.1016/s0731-7085(99)00272-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 458. 30 years of adaptive neural networks: perceptron, Madaline, and backpropagation
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1677
- 年份：1990
- 作者：B. Widrow, M.A. Lehr
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/5.58323
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 459. GaN-Based RF Power Devices and Amplifiers
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1672
- 年份：2008
- 作者：U.K. Mishra, Shen Likun, T.E. Kazior et al.
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/jproc.2007.911060
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 460. Efficient Multi-Scale Attention Module with Cross-Spatial Learning
- 类型：语音/音频
- 标签：Speech, Attention/Transformer
- 重要性：架构论文
- 引用数：1671
- 年份：2023
- 作者：Daliang Ouyang, Su He, Guozhong Zhang et al.
- 来源：ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- 链接：https://doi.org/10.1109/icassp49357.2023.10096516
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 461. Attention-based LSTM for Aspect-level Sentiment Classification
- 类型：深度学习相关
- 标签：Attention/Transformer, RNN
- 重要性：奠基/方法论文
- 引用数：1669
- 年份：2016
- 作者：Yequan Wang, Minlie Huang, xiaoyan zhu et al.
- 来源：Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing
- 链接：https://doi.org/10.18653/v1/d16-1058
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 462. GaN-on-Si Power Technology: Devices and Applications
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1668
- 年份：2017
- 作者：Kevin J. Chen, Oliver Haberlen, Alex Lidow et al.
- 来源：IEEE Transactions on Electron Devices
- 链接：https://doi.org/10.1109/ted.2017.2657579
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 463. Graph Neural Networks for Social Recommendation
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1665
- 年份：2019
- 作者：Wenqi Fan, Yao Ma, Qing Li et al.
- 来源：The World Wide Web Conference
- 链接：https://doi.org/10.1145/3308558.3313488
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 464. Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Residual/Skip Connection, CNN
- 重要性：奠基/方法论文
- 引用数：1665
- 年份：2017
- 作者：Hu Chen, Yi Zhang, Mannudeep K. Kalra et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2017.2715284
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 465. Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：1663
- 年份：2016
- 作者：Gong Cheng, Peicheng Zhou, Junwei Han
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2016.2601622
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 466. GaN: Processing, defects, and devices
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1662
- 年份：1999
- 作者：S. J. Pearton, J. C. Zolper, R. J. Shul et al.
- 来源：Journal of Applied Physics
- 链接：https://doi.org/10.1063/1.371145
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 467. Deep learning for sensor-based activity recognition: A survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1661
- 年份：2019
- 作者：Jindong Wang, Yiqiang Chen, Shuji Hao et al.
- 来源：Pattern Recognition Letters
- 链接：https://doi.org/10.1016/j.patrec.2018.02.010
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 468. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in Vitro
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1659
- 年份：2017
- 作者：Zhedong Zheng, Liang Zheng, Yi Yang
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.405
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 469. Review of Deep Learning Algorithms and Architectures
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1658
- 年份：2019
- 作者：Ajay Shrestha, Ausif Mahmood
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2019.2912200
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 470. Anomalous diffusion models and their properties: non-stationarity, non-ergodicity, and ageing at the centenary of single particle tracking
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：1654
- 年份：2014
- 作者：Ralf Metzler, Jae-Hyung Jeon, Andrey G. Cherstvy et al.
- 来源：Physical Chemistry Chemical Physics
- 链接：https://doi.org/10.1039/c4cp03465a
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 471. Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：1653
- 年份：2018
- 作者：Ryan Poplin, Avinash V. Varadarajan, Katy Blumer et al.
- 来源：Nature Biomedical Engineering
- 链接：https://doi.org/10.1038/s41551-018-0195-0
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 472. Cascade R-CNN: High Quality Object Detection and Instance Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection, Segmentation
- 重要性：架构论文
- 引用数：1652
- 年份：2021
- 作者：Zhaowei Cai, Nuno Vasconcelos
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2019.2956516
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 473. Second-Order Attention Network for Single Image Super-Resolution
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1645
- 年份：2019
- 作者：Tao Dai, Jianrui Cai, Yongbing Zhang et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.01132
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 474. Time series classification from scratch with deep neural networks: A strong baseline
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1643
- 年份：2017
- 作者：Zhiguang Wang, Weizhong Yan, Tim Oates
- 来源：2017 International Joint Conference on Neural Networks (IJCNN)
- 链接：https://doi.org/10.1109/ijcnn.2017.7966039
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 475. Interleukin-2: Inception, Impact, and Implications
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1634
- 年份：1988
- 作者：Kendall A. Smith
- 来源：Science
- 链接：https://doi.org/10.1126/science.3131876
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 476. Deep Learning Approach for Intelligent Intrusion Detection System
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1633
- 年份：2019
- 作者：R. Vinayakumar, Mamoun Alazab, K. P. Soman et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2019.2895334
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 477. LSTM network: a deep learning approach for short‐term traffic forecast
- 类型：系统/框架
- 标签：Framework/System, RNN
- 重要性：系统/框架
- 引用数：1633
- 年份：2017
- 作者：Zheng Zhao, Weihai Chen, Xingming Wu et al.
- 来源：IET Intelligent Transport Systems
- 链接：https://doi.org/10.1049/iet-its.2016.0208
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 478. Person Transfer GAN to Bridge Domain Gap for Person Re-identification
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1631
- 年份：2018
- 作者：Longhui Wei, Shiliang Zhang, Wen Gao et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00016
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 479. A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis
- 类型：医疗/生命科学AI / 系统/框架
- 标签：Medical AI, Framework/System
- 重要性：综述论文
- 引用数：1625
- 年份：2019
- 作者：Xiaoxuan Liu, Livia Faes, Aditya U Kale et al.
- 来源：The Lancet Digital Health
- 链接：https://doi.org/10.1016/s2589-7500(19)30123-2
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 480. Benefits, Limits, and Risks of GPT-4 as an AI Chatbot for Medicine
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：架构论文
- 引用数：1623
- 年份：2023
- 作者：Peter Lee, Sebastien Bubeck, Joseph Petro
- 来源：New England Journal of Medicine
- 链接：https://doi.org/10.1056/nejmsr2214184
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 481. Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：1618
- 年份：2018
- 作者：Naveed Akhtar, Ajmal Mian
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2018.2807385
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 482. RoFormer: Enhanced transformer with Rotary Position Embedding
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：1618
- 年份：2024
- 作者：Jianlin Su, Murtadha Ahmed, Yu Lu et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2023.127063
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 483. Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1615
- 年份：2016
- 作者：Srdjan Sladojevic, Marko Arsenovic, Andras Anderla et al.
- 来源：Computational Intelligence and Neuroscience
- 链接：https://doi.org/10.1155/2016/3289801
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 484. Ensembling neural networks: Many could be better than all
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1614
- 年份：2002
- 作者：Zhi-Hua Zhou, Jianxin Wu, Wei Tang
- 来源：Artificial Intelligence
- 链接：https://doi.org/10.1016/s0004-3702(02)00190-x
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 485. RePaint: Inpainting using Denoising Diffusion Probabilistic Models
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：1610
- 年份：2022
- 作者：Andreas Lugmayr, Martin Danelljan, Andres Romero et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01117
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 486. Deep learning for electroencephalogram (EEG) classification tasks: a review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1608
- 年份：2019
- 作者：Alexander Craik, Yongtian He, Jose L Contreras-Vidal
- 来源：Journal of Neural Engineering
- 链接：https://doi.org/10.1088/1741-2552/ab0ab5
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 487. Remaining useful life estimation in prognostics using deep convolution neural networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1605
- 年份：2018
- 作者：Xiang Li, Qian Ding, Jian-Qiao Sun
- 来源：Reliability Engineering &amp;amp; System Safety
- 链接：https://doi.org/10.1016/j.ress.2017.11.021
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 488. Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1602
- 年份：2015
- 作者：Chen Zhang, Peng Li, Guangyu Sun et al.
- 来源：Proceedings of the 2015 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays
- 链接：https://doi.org/10.1145/2684746.2689060
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 489. Graph Convolutional Networks for Hyperspectral Image Classification
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：架构论文
- 引用数：1602
- 年份：2021
- 作者：Danfeng Hong, Lianru Gao, Jing Yao et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2020.3015157
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 490. Planning chemical syntheses with deep neural networks and symbolic AI
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1593
- 年份：2018
- 作者：Marwin H. S. Segler, Mike Preuss, Mark P. Waller
- 来源：Nature
- 链接：https://doi.org/10.1038/nature25978
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 491. Deep Learning for Health Informatics
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：1592
- 年份：2017
- 作者：Daniele Ravì, Charence Wong, Fani Deligianni et al.
- 来源：IEEE Journal of Biomedical and Health Informatics
- 链接：https://doi.org/10.1109/jbhi.2016.2636665
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 492. Deep Learning Classification of Land Cover and Crop Types Using Remote Sensing Data
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1585
- 年份：2017
- 作者：Nataliia Kussul, Mykola Lavreniuk, Sergii Skakun et al.
- 来源：IEEE Geoscience and Remote Sensing Letters
- 链接：https://doi.org/10.1109/lgrs.2017.2681128
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 493. A Jump-Diffusion Model for Option Pricing
- 类型：生成模型
- 标签：Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：1582
- 年份：2002
- 作者：S. G. Kou
- 来源：Management Science
- 链接：https://doi.org/10.1287/mnsc.48.8.1086.166
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 494. A survey of deep learning techniques for autonomous driving
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1581
- 年份：2020
- 作者：Sorin Grigorescu, Bogdan Trasnea, Tiberiu Cocias et al.
- 来源：Journal of Field Robotics
- 链接：https://doi.org/10.1002/rob.21918
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 495. A Deep Learning Approach to Network Intrusion Detection
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1579
- 年份：2018
- 作者：Nathan Shone, Tran Nguyen Ngoc, Vu Dinh Phai et al.
- 来源：IEEE Transactions on Emerging Topics in Computational Intelligence
- 链接：https://doi.org/10.1109/tetci.2017.2772792
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 496. InceptionTime: Finding AlexNet for time series classification
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1579
- 年份：2020
- 作者：Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier et al.
- 来源：Data Mining and Knowledge Discovery
- 链接：https://doi.org/10.1007/s10618-020-00710-y
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 497. Deep learning in environmental remote sensing: Achievements and challenges
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1564
- 年份：2020
- 作者：Qiangqiang Yuan, Huanfeng Shen, Tongwen Li et al.
- 来源：Remote Sensing of Environment
- 链接：https://doi.org/10.1016/j.rse.2020.111716
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 498. Efficient Memory Management for Large Language Model Serving with PagedAttention
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System, Attention/Transformer
- 重要性：系统/框架
- 引用数：1562
- 年份：2023
- 作者：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang et al.
- 来源：Proceedings of the 29th Symposium on Operating Systems Principles
- 链接：https://doi.org/10.1145/3600006.3613165
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 499. Generative adversarial network in medical imaging: A review
- 类型：计算机视觉 / 生成模型 / 医疗/生命科学AI
- 标签：视觉, Generative AI, Medical AI, GAN
- 重要性：综述论文
- 引用数：1559
- 年份：2019
- 作者：Xin Yi, Ekta Walia, Paul Babyn
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2019.101552
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 500. Reaction-Diffusion Model as a Framework for Understanding Biological Pattern Formation
- 类型：生成模型 / 系统/框架
- 标签：Generative AI, Framework/System, Diffusion/Flow
- 重要性：系统/框架
- 引用数：1557
- 年份：2010
- 作者：Shigeru Kondo, Takashi Miura
- 来源：Science
- 链接：https://doi.org/10.1126/science.1179047
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 501. Multimodal Transformer for Unaligned Multimodal Language Sequences
- 类型：NLP/语言模型 / 多模态
- 标签：Language Model, Multimodal, Attention/Transformer
- 重要性：架构论文
- 引用数：1556
- 年份：2019
- 作者：Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang et al.
- 来源：Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics
- 链接：https://doi.org/10.18653/v1/p19-1656
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 502. FFA-Net: Feature Fusion Attention Network for Single Image Dehazing
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1555
- 年份：2020
- 作者：Xu Qin, Zhilin Wang, Yuanchao Bai et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v34i07.6865
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 503. 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN, MoE/Sparse, Segmentation
- 重要性：架构论文
- 引用数：1550
- 年份：2018
- 作者：Benjamin Graham, Martin Engelcke, Laurens van der Maaten
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00961
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 504. A survey on large language model based autonomous agents
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：综述论文
- 引用数：1550
- 年份：2024
- 作者：Lei Wang, Chen Ma, Xueyang Feng et al.
- 来源：Frontiers of Computer Science
- 链接：https://doi.org/10.1007/s11704-024-40231-1
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 505. Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection
- 类型：计算机视觉
- 标签：视觉, Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：1546
- 年份：2019
- 作者：Dong Gong, Lingqiao Liu, Vuong Le et al.
- 来源：2019 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2019.00179
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 506. PointConv: Deep Convolutional Networks on 3D Point Clouds
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1545
- 年份：2019
- 作者：Wenxuan Wu, Zhongang Qi, Li Fuxin
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00985
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 507. Deep neural networks: A promising tool for fault characteristic mining and intelligent diagnosis of rotating machinery with massive data
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1540
- 年份：2016
- 作者：Feng Jia, Yaguo Lei, Jing Lin et al.
- 来源：Mechanical Systems and Signal Processing
- 链接：https://doi.org/10.1016/j.ymssp.2015.10.025
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 508. ChloroP, a neural network‐based method for predicting chloroplast transit peptides and their cleavage sites
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1538
- 年份：1999
- 作者：Olof Emanuelsson, Henrik Nielsen, Gunnar Von Heijne
- 来源：Protein Science
- 链接：https://doi.org/10.1110/ps.8.5.978
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 509. Robust Physical-World Attacks on Deep Learning Visual Classification
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1534
- 年份：2018
- 作者：Kevin Eykholt, Ivan Evtimov, Earlence Fernandes et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00175
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 510. Deep Learning Enabled Semantic Communication Systems
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1534
- 年份：2021
- 作者：Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li et al.
- 来源：IEEE Transactions on Signal Processing
- 链接：https://doi.org/10.1109/tsp.2021.3071210
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 511. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1534
- 年份：2019
- 作者：Ya Su, Youjian Zhao, Chenhao Niu et al.
- 来源：Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp;amp; Data Mining
- 链接：https://doi.org/10.1145/3292500.3330672
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 512. Prediction of continuous B‐cell epitopes in an antigen using recurrent neural network
- 类型：医疗/生命科学AI
- 标签：Medical AI, RNN
- 重要性：架构论文
- 引用数：1534
- 年份：2006
- 作者：Sudipto Saha, G. P. S. Raghava
- 来源：Proteins: Structure, Function, and Bioinformatics
- 链接：https://doi.org/10.1002/prot.21078
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 513. Two dimensional electron gases induced by spontaneous and piezoelectric polarization in undoped and doped AlGaN/GaN heterostructures
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1534
- 年份：2000
- 作者：O. Ambacher, B. Foutz, J. Smart et al.
- 来源：Journal of Applied Physics
- 链接：https://doi.org/10.1063/1.371866
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 514. Palette: Image-to-Image Diffusion Models
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Diffusion/Flow
- 重要性：应用/方法论文
- 引用数：1530
- 年份：2022
- 作者：Chitwan Saharia, William Chan, Huiwen Chang et al.
- 来源：Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings
- 链接：https://doi.org/10.1145/3528233.3530757
- 概述：研究扩散/score-based生成模型，通过逐步去噪实现高质量生成。

## 515. The rise of deep learning in drug discovery
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：1529
- 年份：2018
- 作者：Hongming Chen, Ola Engkvist, Yinhai Wang et al.
- 来源：Drug Discovery Today
- 链接：https://doi.org/10.1016/j.drudis.2018.01.039
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 516. Deep Learning for Image Super-Resolution: A Survey
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：1526
- 年份：2021
- 作者：Zhihao Wang, Jian Chen, Steven C. H. Hoi
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2020.2982166
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 517. SCA-CNN: Spatial and Channel-Wise Attention in Convolutional Networks for Image Captioning
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：1522
- 年份：2017
- 作者：Long Chen, Hanwang Zhang, Jun Xiao et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.667
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 518. Self-Attention with Relative Position Representations
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：1516
- 年份：2018
- 作者：Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
- 来源：Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)
- 链接：https://doi.org/10.18653/v1/n18-2074
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 519. Time-series forecasting with deep learning: a survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1514
- 年份：2021
- 作者：Bryan Lim, Stefan Zohren
- 来源：Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences
- 链接：https://doi.org/10.1098/rsta.2020.0209
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 520. LXMERT: Learning Cross-Modality Encoder Representations from Transformers
- 类型：NLP/语言模型 / 多模态
- 标签：Language Model, Multimodal, Attention/Transformer
- 重要性：架构论文
- 引用数：1514
- 年份：2019
- 作者：Hao Tan, Mohit Bansal
- 来源：Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)
- 链接：https://doi.org/10.18653/v1/d19-1514
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 521. PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1513
- 年份：2018
- 作者：Yu Xiang, Tanner Schmidt, Venkatraman Narayanan et al.
- 来源：Robotics: Science and Systems XIV
- 链接：https://doi.org/10.15607/rss.2018.xiv.019
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 522. Learning RoI Transformer for Oriented Object Detection in Aerial Images
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Object Detection
- 重要性：架构论文
- 引用数：1510
- 年份：2019
- 作者：Jian Ding, Nan Xue, Yang Long et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00296
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 523. Session-Based Recommendation with Graph Neural Networks
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1509
- 年份：2019
- 作者：Shu Wu, Yuyuan Tang, Yanqiao Zhu et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v33i01.3301346
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 524. Deep Learning at Chest Radiography: Automated Classification of Pulmonary Tuberculosis by Using Convolutional Neural Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：1502
- 年份：2017
- 作者：Paras Lakhani, Baskaran Sundaram
- 来源：Radiology
- 链接：https://doi.org/10.1148/radiol.2017162326
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 525. Libra R-CNN: Towards Balanced Learning for Object Detection
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：1502
- 年份：2019
- 作者：Jiangmiao Pang, Kai Chen, Jianping Shi et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00091
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 526. Deep learning for smart manufacturing: Methods and applications
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1502
- 年份：2018
- 作者：Jinjiang Wang, Yulin Ma, Laibin Zhang et al.
- 来源：Journal of Manufacturing Systems
- 链接：https://doi.org/10.1016/j.jmsy.2018.01.003
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 527. Neural networks for control systems—A survey
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1502
- 年份：1992
- 作者：K.J. Hunt, D. Sbarbaro, R. Żbikowski et al.
- 来源：Automatica
- 链接：https://doi.org/10.1016/0005-1098(92)90053-i
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 528. Self-Supervised Visual Feature Learning With Deep Neural Networks: A Survey
- 类型：计算机视觉
- 标签：视觉, Self-supervised Learning
- 重要性：综述论文
- 引用数：1500
- 年份：2021
- 作者：Longlong Jing, Yingli Tian
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2020.2992393
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 529. A language modeling approach to information retrieval
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：1498
- 年份：1998
- 作者：Jay M. Ponte, W. Bruce Croft
- 来源：Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval
- 链接：https://doi.org/10.1145/290941.291008
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 530. Targeted Synthesis of a Porous Aromatic Framework with High Stability and Exceptionally High Surface Area
- 类型：计算机视觉 / 生成模型 / 系统/框架
- 标签：视觉, Generative AI, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1498
- 年份：2009
- 作者：Teng Ben, Hao Ren, Shengqian Ma et al.
- 来源：Angewandte Chemie International Edition
- 链接：https://doi.org/10.1002/anie.200904637
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 531. A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1497
- 年份：2017
- 作者：Wei Zhang, Gaoliang Peng, Chuanhao Li et al.
- 来源：Sensors
- 链接：https://doi.org/10.3390/s17020425
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 532. Amyloid-β–induced neuronal dysfunction in Alzheimer's disease: from synapses toward neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1497
- 年份：2010
- 作者：Jorge J Palop, Lennart Mucke
- 来源：Nature Neuroscience
- 链接：https://doi.org/10.1038/nn.2583
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 533. Fuzzy ARTMAP: A neural network architecture for incremental supervised learning of analog multidimensional maps
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1492
- 年份：1992
- 作者：G.A. Carpenter, S. Grossberg, N. Markuzon et al.
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.159059
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 534. Soft robotic glove for combined assistance and at-home rehabilitation
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System, Word Embedding
- 重要性：系统/框架
- 引用数：1488
- 年份：2015
- 作者：Panagiotis Polygerinos, Zheng Wang, Kevin C. Galloway et al.
- 来源：Robotics and Autonomous Systems
- 链接：https://doi.org/10.1016/j.robot.2014.08.014
- 概述：学习词向量/分布式语义表示，为神经NLP模型提供基础表征。

## 535. State-of-the-Art in Visual Attention Modeling
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1487
- 年份：2013
- 作者：Ali Borji, Laurent Itti
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2012.89
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 536. IFCNN: A general image fusion framework based on convolutional neural network
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1480
- 年份：2020
- 作者：Yu Zhang, Yu Liu, Peng Sun et al.
- 来源：Information Fusion
- 链接：https://doi.org/10.1016/j.inffus.2019.07.011
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 537. GMAN: A Graph Multi-Attention Network for Traffic Prediction
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：1479
- 年份：2020
- 作者：Chuanpan Zheng, Xiaoliang Fan, Cheng Wang et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v34i01.5477
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 538. Over-the-Air Deep Learning Based Radio Signal Classification
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1478
- 年份：2018
- 作者：Timothy James O'Shea, Tamoghna Roy, T. Charles Clancy
- 来源：IEEE Journal of Selected Topics in Signal Processing
- 链接：https://doi.org/10.1109/jstsp.2018.2797022
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 539. Using goal-driven deep learning models to understand sensory cortex
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1477
- 年份：2016
- 作者：Daniel L K Yamins, James J DiCarlo
- 来源：Nature Neuroscience
- 链接：https://doi.org/10.1038/nn.4244
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 540. Deep Learning in Mobile and Wireless Networking: A Survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1476
- 年份：2019
- 作者：Chaoyun Zhang, Paul Patras, Hamed Haddadi
- 来源：IEEE Communications Surveys &amp;amp; Tutorials
- 链接：https://doi.org/10.1109/comst.2019.2904897
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 541. Long short-term memory recurrent neural network architectures for large scale acoustic modeling
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：1472
- 年份：2014
- 作者：Haşim Sak, Andrew Senior, Françoise Beaufays
- 来源：Interspeech 2014
- 链接：https://doi.org/10.21437/interspeech.2014-80
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 542. A Survey of the Usages of Deep Learning for Natural Language Processing
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1468
- 年份：2021
- 作者：Daniel W. Otter, Julian R. Medina, Jugal K. Kalita
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2020.2979670
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 543. Deep convolutional neural network for the automated detection and diagnosis of seizure using EEG signals
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1465
- 年份：2018
- 作者：U. Rajendra Acharya, Shu Lih Oh, Yuki Hagiwara et al.
- 来源：Computers in Biology and Medicine
- 链接：https://doi.org/10.1016/j.compbiomed.2017.09.017
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 544. Atom-centered symmetry functions for constructing high-dimensional neural network potentials
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1464
- 年份：2011
- 作者：Jörg Behler
- 来源：The Journal of Chemical Physics
- 链接：https://doi.org/10.1063/1.3553717
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 545. Graph Convolutional Networks for Text Classification
- 类型：NLP/语言模型 / 计算机视觉 / 图学习
- 标签：Language Model, 视觉, Graph, CNN
- 重要性：架构论文
- 引用数：1462
- 年份：2019
- 作者：Liang Yao, Chengsheng Mao, Yuan Luo
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v33i01.33017370
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 546. Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks
- 类型：深度学习相关
- 标签：Optimization/Training
- 重要性：架构论文
- 引用数：1458
- 年份：2019
- 作者：Emre O. Neftci, Hesham Mostafa, Friedemann Zenke
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/msp.2019.2931595
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 547. Hypergraph Neural Networks
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1452
- 年份：2019
- 作者：Yifan Feng, Haoxuan You, Zizhao Zhang et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v33i01.33013558
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 548. Advantages and disadvantages of using artificial neural networks versus logistic regression for predicting medical outcomes
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1452
- 年份：1996
- 作者：Jack V. Tu
- 来源：Journal of Clinical Epidemiology
- 链接：https://doi.org/10.1016/s0895-4356(96)00002-9
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 549. SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1451
- 年份：2022
- 作者：Jiayi Ma, Linfeng Tang, Fan Fan et al.
- 来源：IEEE/CAA Journal of Automatica Sinica
- 链接：https://doi.org/10.1109/jas.2022.105686
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 550. Guest Editorial Deep Learning in Medical Imaging: Overview and Future Promise of an Exciting New Technique
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：综述论文
- 引用数：1450
- 年份：2016
- 作者：Hayit Greenspan, Bram van Ginneken, Ronald M. Summers
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2553401
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 551. Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：应用/方法论文
- 引用数：1447
- 年份：2019
- 作者：Mohammad Hesam Hesamian, Wenjing Jia, Xiangjian He et al.
- 来源：Journal of Digital Imaging
- 链接：https://doi.org/10.1007/s10278-019-00227-x
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 552. Self-supervised Graph Learning for Recommendation
- 类型：深度学习相关
- 标签：Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：1446
- 年份：2021
- 作者：Jiancan Wu, Xiang Wang, Fuli Feng et al.
- 来源：Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval
- 链接：https://doi.org/10.1145/3404835.3462862
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 553. N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：架构论文
- 引用数：1444
- 年份：2018
- 作者：Yair Meidan, Michael Bohadana, Yael Mathov et al.
- 来源：IEEE Pervasive Computing
- 链接：https://doi.org/10.1109/mprv.2018.03367731
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 554. Low-Dose CT Image Denoising Using a Generative Adversarial Network With Wasserstein Distance and Perceptual Loss
- 类型：计算机视觉 / 生成模型 / 医疗/生命科学AI
- 标签：视觉, Generative AI, Medical AI, GAN
- 重要性：架构论文
- 引用数：1437
- 年份：2018
- 作者：Qingsong Yang, Pingkun Yan, Yanbo Zhang et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2018.2827462
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 555. Convolutional networks and applications in vision
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1433
- 年份：2010
- 作者：Yann LeCun, Koray Kavukcuoglu, Clement Farabet
- 来源：Proceedings of 2010 IEEE International Symposium on Circuits and Systems
- 链接：https://doi.org/10.1109/iscas.2010.5537907
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 556. Privacy-Preserving Deep Learning
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1431
- 年份：2015
- 作者：Reza Shokri, Vitaly Shmatikov
- 来源：Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security
- 链接：https://doi.org/10.1145/2810103.2813687
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 557. pixelNeRF: Neural Radiance Fields from One or Few Images
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1431
- 年份：2021
- 作者：Alex Yu, Vickie Ye, Matthew Tancik et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00455
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 558. A Survey on Contrastive Self-Supervised Learning
- 类型：深度学习相关
- 标签：Self-supervised Learning
- 重要性：综述论文
- 引用数：1430
- 年份：2020
- 作者：Ashish Jaiswal, Ashwin Ramesh Babu, Mohammad Zaki Zadeh et al.
- 来源：Technologies
- 链接：https://doi.org/10.3390/technologies9010002
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 559. Solving high-dimensional partial differential equations using deep learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1429
- 年份：2018
- 作者：Jiequn Han, Arnulf Jentzen, Weinan E
- 来源：Proceedings of the National Academy of Sciences
- 链接：https://doi.org/10.1073/pnas.1718942115
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 560. DeepEMhancer: a deep learning solution for cryo-EM volume post-processing
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1428
- 年份：2021
- 作者：Ruben Sanchez-Garcia, Josue Gomez-Blanco, Ana Cuervo et al.
- 来源：Communications Biology
- 链接：https://doi.org/10.1038/s42003-021-02399-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 561. Deep Learning--based Text Classification
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：综述论文
- 引用数：1426
- 年份：2022
- 作者：Shervin Minaee, Nal Kalchbrenner, Erik Cambria et al.
- 来源：ACM Computing Surveys
- 链接：https://doi.org/10.1145/3439726
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 562. Autonomous Structural Visual Inspection Using Region‐Based Deep Learning for Detecting Multiple Damage Types
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1425
- 年份：2018
- 作者：Young‐Jin Cha, Wooram Choi, Gahyun Suh et al.
- 来源：Computer-Aided Civil and Infrastructure Engineering
- 链接：https://doi.org/10.1111/mice.12334
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 563. Transformer Tracking
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1424
- 年份：2021
- 作者：Xin Chen, Bin Yan, Jiawen Zhu et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00803
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 564. Reynolds averaged turbulence modelling using deep neural networks with embedded invariance
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1422
- 年份：2016
- 作者：Julia Ling, Andrew Kurzawski, Jeremy Templeton
- 来源：Journal of Fluid Mechanics
- 链接：https://doi.org/10.1017/jfm.2016.615
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 565. EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：架构论文
- 引用数：1418
- 年份：2020
- 作者：Tengfei Song, Wenming Zheng, Peng Song et al.
- 来源：IEEE Transactions on Affective Computing
- 链接：https://doi.org/10.1109/taffc.2018.2817622
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 566. AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, Attention/Transformer, GAN
- 重要性：架构论文
- 引用数：1413
- 年份：2018
- 作者：Tao Xu, Pengchuan Zhang, Qiuyuan Huang et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00143
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 567. Learning IoT in Edge: Deep Learning for the Internet of Things with Edge Computing
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1411
- 年份：2018
- 作者：He Li, Kaoru Ota, Mianxiong Dong
- 来源：IEEE Network
- 链接：https://doi.org/10.1109/mnet.2018.1700202
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 568. Object detection using YOLO: challenges, architectural successors, datasets and applications
- 类型：计算机视觉
- 标签：视觉, Dataset/Benchmark, Object Detection
- 重要性：数据集/基准
- 引用数：1408
- 年份：2023
- 作者：Tausif Diwan, G. Anirudh, Jitendra V. Tembhurne
- 来源：Multimedia Tools and Applications
- 链接：https://doi.org/10.1007/s11042-022-13644-y
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 569. SIGNATURE VERIFICATION USING A “SIAMESE” TIME DELAY NEURAL NETWORK
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1406
- 年份：1993
- 作者：JANE BROMLEY, JAMES W. BENTZ, LÉON BOTTOU et al.
- 来源：International Journal of Pattern Recognition and Artificial Intelligence
- 链接：https://doi.org/10.1142/s0218001493000339
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 570. The Performance of LSTM and BiLSTM in Forecasting Time Series
- 类型：深度学习相关
- 标签：RNN
- 重要性：奠基/方法论文
- 引用数：1400
- 年份：2019
- 作者：Sima Siami-Namini, Neda Tavakoli, Akbar Siami Namin
- 来源：2019 IEEE International Conference on Big Data (Big Data)
- 链接：https://doi.org/10.1109/bigdata47090.2019.9005997
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 571. An Empirical Study of Training Self-Supervised Vision Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Self-supervised Learning
- 重要性：架构论文
- 引用数：1399
- 年份：2021
- 作者：Xinlei Chen, Saining Xie, Kaiming He
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00950
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 572. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1396
- 年份：2009
- 作者：Honglak Lee, Roger Grosse, Rajesh Ranganath et al.
- 来源：Proceedings of the 26th Annual International Conference on Machine Learning
- 链接：https://doi.org/10.1145/1553374.1553453
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 573. Extreme learning machine: a new learning scheme of feedforward neural networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1396
- 年份：2005
- 作者：Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew
- 来源：2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No.04CH37541)
- 链接：https://doi.org/10.1109/ijcnn.2004.1380068
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 574. f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks
- 类型：计算机视觉 / 生成模型 / 医疗/生命科学AI
- 标签：视觉, Generative AI, Medical AI, GAN
- 重要性：架构论文
- 引用数：1395
- 年份：2019
- 作者：Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2019.01.010
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 575. Deep Residual Shrinkage Networks for Fault Diagnosis
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：1390
- 年份：2020
- 作者：Minghang Zhao, Shisheng Zhong, Xuyun Fu et al.
- 来源：IEEE Transactions on Industrial Informatics
- 链接：https://doi.org/10.1109/tii.2019.2943898
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 576. Predicting residential energy consumption using CNN-LSTM neural networks
- 类型：计算机视觉
- 标签：视觉, CNN, RNN
- 重要性：奠基/方法论文
- 引用数：1390
- 年份：2019
- 作者：Tae-Young Kim, Sung-Bae Cho
- 来源：Energy
- 链接：https://doi.org/10.1016/j.energy.2019.05.230
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 577. Deep Learning With Edge Computing: A Review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1386
- 年份：2019
- 作者：Jiasi Chen, Xukan Ran
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/jproc.2019.2921977
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 578. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1385
- 年份：2018
- 作者：John R. Zech, Marcus A. Badgeley, Manway Liu et al.
- 来源：PLOS Medicine
- 链接：https://doi.org/10.1371/journal.pmed.1002683
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 579. Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting
- 类型：计算机视觉 / 图学习 / 系统/框架
- 标签：视觉, Graph, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1385
- 年份：2020
- 作者：Chao Song, Youfang Lin, Shengnan Guo et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v34i01.5438
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 580. Best practices for convolutional neural networks applied to visual document analysis
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, CNN
- 重要性：架构论文
- 引用数：1384
- 年份：2005
- 作者：P.Y. Simard, D. Steinkraus, J.C. Platt
- 来源：Seventh International Conference on Document Analysis and Recognition, 2003. Proceedings.
- 链接：https://doi.org/10.1109/icdar.2003.1227801
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 581. Deep Learning Face Representation from Predicting 10,000 Classes
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1380
- 年份：2014
- 作者：Yi Sun, Xiaogang Wang, Xiaoou Tang
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2014.244
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 582. Accurate medium-range global weather forecasting with 3D neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1379
- 年份：2023
- 作者：Kaifeng Bi, Lingxi Xie, Hengheng Zhang et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-023-06185-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 583. Quantum convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1377
- 年份：2019
- 作者：Iris Cong, Soonwon Choi, Mikhail D. Lukin
- 来源：Nature Physics
- 链接：https://doi.org/10.1038/s41567-019-0648-8
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 584. A survey on deep learning techniques for image and video semantic segmentation
- 类型：NLP/语言模型 / 计算机视觉 / 视频/世界模型
- 标签：Language Model, 视觉, Video, Attention/Transformer, Segmentation
- 重要性：综述论文
- 引用数：1376
- 年份：2018
- 作者：Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea et al.
- 来源：Applied Soft Computing
- 链接：https://doi.org/10.1016/j.asoc.2018.05.018
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 585. Domain Adaptive Faster R-CNN for Object Detection in the Wild
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：1376
- 年份：2018
- 作者：Yuhua Chen, Wen Li, Christos Sakaridis et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00352
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 586. A survey on deep learning and its applications
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1370
- 年份：2021
- 作者：Shi Dong, Ping Wang, Khushnood Abbas
- 来源：Computer Science Review
- 链接：https://doi.org/10.1016/j.cosrev.2021.100379
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 587. Deep learning for computational biology
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1367
- 年份：2016
- 作者：Christof Angermueller, Tanel Pärnamaa, Leopold Parts et al.
- 来源：Molecular Systems Biology
- 链接：https://doi.org/10.15252/msb.20156651
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 588. Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：综述论文
- 引用数：1366
- 年份：2018
- 作者：Benjamin Shickel, Patrick James Tighe, Azra Bihorac et al.
- 来源：IEEE Journal of Biomedical and Health Informatics
- 链接：https://doi.org/10.1109/jbhi.2017.2767063
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 589. Wider or Deeper: Revisiting the ResNet Model for Visual Recognition
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：架构论文
- 引用数：1364
- 年份：2019
- 作者：Zifeng Wu, Chunhua Shen, Anton van den Hengel
- 来源：Pattern Recognition
- 链接：https://doi.org/10.1016/j.patcog.2019.01.006
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 590. DeepTMHMM predicts alpha and beta transmembrane proteins using deep neural networks
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1358
- 年份：2022
- 作者：Jeppe Hallgren, Konstantinos D. Tsirigos, Mads Damgaard Pedersen et al.
- 来源：—
- 链接：https://doi.org/10.1101/2022.04.08.487609
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 591. Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：1357
- 年份：2019
- 作者：Jakob Nikolas Kather, Alexander T. Pearson, Niels Halama et al.
- 来源：Nature Medicine
- 链接：https://doi.org/10.1038/s41591-019-0462-y
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 592. Deep Learning Based Recommender System
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1353
- 年份：2020
- 作者：Shuai Zhang, Lina Yao, Aixin Sun et al.
- 来源：ACM Computing Surveys
- 链接：https://doi.org/10.1145/3285029
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 593. Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing
- 类型：NLP/语言模型 / 医疗/生命科学AI
- 标签：Language Model, Medical AI, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：1352
- 年份：2022
- 作者：Yu Gu, Robert Tinn, Hao Cheng et al.
- 来源：ACM Transactions on Computing for Healthcare
- 链接：https://doi.org/10.1145/3458754
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 594. Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1350
- 年份：2018
- 作者：H.A. Haenssle, C. Fink, R. Schneiderbauer et al.
- 来源：Annals of Oncology
- 链接：https://doi.org/10.1093/annonc/mdy166
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 595. Deep learning-based electroencephalography analysis: a systematic review
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System, Attention/Transformer
- 重要性：综述论文
- 引用数：1350
- 年份：2019
- 作者：Yannick Roy, Hubert Banville, Isabela Albuquerque et al.
- 来源：Journal of Neural Engineering
- 链接：https://doi.org/10.1088/1741-2552/ab260c
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 596. A State-of-the-Art Survey on Deep Learning Theory and Architectures
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1349
- 年份：2019
- 作者：Md Zahangir Alom, Tarek M. Taha, Chris Yakopcic et al.
- 来源：Electronics
- 链接：https://doi.org/10.3390/electronics8030292
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 597. Training Language Models to Follow Instructions with Human Feedback
- 类型：NLP/语言模型 / 系统/框架
- 标签：Language Model, Framework/System
- 重要性：系统/框架
- 引用数：1348
- 年份：2022
- 作者：Long Ouyang, Jeffrey Wu, Xu Jiang et al.
- 来源：Advances in Neural Information Processing Systems 35
- 链接：https://doi.org/10.52202/068431-2011
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 598. PCANet: A Simple Deep Learning Baseline for Image Classification?
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1346
- 年份：2015
- 作者：Tsung-Han Chan, Kui Jia, Shenghua Gao et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2015.2475625
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 599. Scaling deep learning for materials discovery
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1345
- 年份：2023
- 作者：Amil Merchant, Simon Batzner, Samuel S. Schoenholz et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-023-06735-9
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 600. Fully Convolutional Siamese Networks for Change Detection
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：1344
- 年份：2018
- 作者：Rodrigo Caye Daudt, Bertr Le Saux, Alexandre Boulch
- 来源：2018 25th IEEE International Conference on Image Processing (ICIP)
- 链接：https://doi.org/10.1109/icip.2018.8451652
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 601. Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1343
- 年份：2019
- 作者：Milad Nasr, Reza Shokri, Amir Houmansadr
- 来源：2019 IEEE Symposium on Security and Privacy (SP)
- 链接：https://doi.org/10.1109/sp.2019.00065
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 602. Deep learning enables rapid identification of potent DDR1 kinase inhibitors
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1341
- 年份：2019
- 作者：Alex Zhavoronkov, Yan A. Ivanenkov, Alex Aliper et al.
- 来源：Nature Biotechnology
- 链接：https://doi.org/10.1038/s41587-019-0224-x
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 603. Artificial intelligence and deep learning in ophthalmology
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1339
- 年份：2019
- 作者：Daniel Shu Wei Ting, Louis R Pasquale, Lily Peng et al.
- 来源：British Journal of Ophthalmology
- 链接：https://doi.org/10.1136/bjophthalmol-2018-313173
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 604. Stacked Attention Networks for Image Question Answering
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1338
- 年份：2016
- 作者：Zichao Yang, Xiaodong He, Jianfeng Gao et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.10
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 605. Rate of progression of mild cognitive impairment to dementia – meta‐analysis of 41 robust inception cohort studies
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1337
- 年份：2009
- 作者：A. J. Mitchell, M. Shiri‐Feshki
- 来源：Acta Psychiatrica Scandinavica
- 链接：https://doi.org/10.1111/j.1600-0447.2008.01326.x
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 606. A review of vibration-based damage detection in civil structures: From traditional methods to Machine Learning and Deep Learning applications
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1330
- 年份：2021
- 作者：Onur Avci, Osama Abdeljaber, Serkan Kiranyaz et al.
- 来源：Mechanical Systems and Signal Processing
- 链接：https://doi.org/10.1016/j.ymssp.2020.107077
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 607. Deep Recurrent Neural Networks for Hyperspectral Image Classification
- 类型：计算机视觉
- 标签：视觉, RNN
- 重要性：架构论文
- 引用数：1328
- 年份：2017
- 作者：Lichao Mou, Pedram Ghamisi, Xiao Xiang Zhu
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2016.2636241
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 608. Introduction to multi-layer feed-forward neural networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1328
- 年份：1997
- 作者：Daniel Svozil, Vladimír Kvasnicka, Jir̂í Pospichal
- 来源：Chemometrics and Intelligent Laboratory Systems
- 链接：https://doi.org/10.1016/s0169-7439(97)00061-0
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 609. UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：1327
- 年份：2022
- 作者：Libo Wang, Rui Li, Ce Zhang et al.
- 来源：ISPRS Journal of Photogrammetry and Remote Sensing
- 链接：https://doi.org/10.1016/j.isprsjprs.2022.06.008
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 610. Plenoxels: Radiance Fields without Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1326
- 年份：2022
- 作者：Sara Fridovich-Keil, Alex Yu, Matthew Tancik et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00542
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 611. Listen, attend and spell: A neural network for large vocabulary conversational speech recognition
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：1324
- 年份：2016
- 作者：William Chan, Navdeep Jaitly, Quoc Le et al.
- 来源：2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- 链接：https://doi.org/10.1109/icassp.2016.7472621
- 概述：研究深度语音/音频建模，服务于识别、分离或表征学习任务。

## 612. Artificial intelligence to deep learning: machine intelligence approach for drug discovery
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：1321
- 年份：2021
- 作者：Rohan Gupta, Devesh Srivastava, Mehar Sahu et al.
- 来源：Molecular Diversity
- 链接：https://doi.org/10.1007/s11030-021-10217-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 613. CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1320
- 年份：2018
- 作者：Yuhong Li, Xiaofan Zhang, Deming Chen
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00120
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 614. Nonlinear neural networks: Principles, mechanisms, and architectures
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1319
- 年份：1988
- 作者：Stephen Grossberg
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/0893-6080(88)90021-4
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 615. Adversarial Examples: Attacks and Defenses for Deep Learning
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1316
- 年份：2019
- 作者：Xiaoyong Yuan, Pan He, Qile Zhu et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2018.2886017
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 616. ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1315
- 年份：2017
- 作者：Jian-Hao Luo, Jianxin Wu, Weiyao Lin
- 来源：2017 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2017.541
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 617. Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1313
- 年份：2017
- 作者：Justin Salamon, Juan Pablo Bello
- 来源：IEEE Signal Processing Letters
- 链接：https://doi.org/10.1109/lsp.2017.2657381
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 618. More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification
- 类型：计算机视觉 / 多模态
- 标签：视觉, Multimodal
- 重要性：应用/方法论文
- 引用数：1313
- 年份：2021
- 作者：Danfeng Hong, Lianru Gao, Naoto Yokoya et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2020.3016820
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 619. Deep learning for detecting robotic grasps
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1313
- 年份：2015
- 作者：Ian Lenz, Honglak Lee, Ashutosh Saxena
- 来源：The International Journal of Robotics Research
- 链接：https://doi.org/10.1177/0278364914549607
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 620. Real-Time Motor Fault Detection by 1-D Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1313
- 年份：2016
- 作者：Turker Ince, Serkan Kiranyaz, Levent Eren et al.
- 来源：IEEE Transactions on Industrial Electronics
- 链接：https://doi.org/10.1109/tie.2016.2582729
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 621. DDcGAN: A Dual-Discriminator Conditional Generative Adversarial Network for Multi-Resolution Image Fusion
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：1312
- 年份：2020
- 作者：Jiayi Ma, Han Xu, Junjun Jiang et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2020.2977573
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 622. Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
- 类型：计算机视觉
- 标签：视觉, CNN, MoE/Sparse, Object Detection
- 重要性：架构论文
- 引用数：1309
- 年份：2021
- 作者：Peize Sun, Rufeng Zhang, Yi Jiang et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.01422
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 623. A deep convolutional neural network with new training methods for bearing fault diagnosis under noisy environment and different working load
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1308
- 年份：2018
- 作者：Wei Zhang, Chuanhao Li, Gaoliang Peng et al.
- 来源：Mechanical Systems and Signal Processing
- 链接：https://doi.org/10.1016/j.ymssp.2017.06.022
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 624. Large Kernel Matters — Improve Semantic Segmentation by Global Convolutional Network
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：1307
- 年份：2017
- 作者：Chao Peng, Xiangyu Zhang, Gang Yu et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.189
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 625. Oriented R-CNN for Object Detection
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：1301
- 年份：2021
- 作者：Xingxing Xie, Gong Cheng, Jiabao Wang et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00350
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 626. Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1300
- 年份：2021
- 作者：Wojciech Samek, Gregoire Montavon, Sebastian Lapuschkin et al.
- 来源：Proceedings of the IEEE
- 链接：https://doi.org/10.1109/jproc.2021.3060483
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 627. Convergence of Edge Computing and Deep Learning: A Comprehensive Survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1300
- 年份：2020
- 作者：Xiaofei Wang, Yiwen Han, Victor C. M. Leung et al.
- 来源：IEEE Communications Surveys &amp;amp; Tutorials
- 链接：https://doi.org/10.1109/comst.2020.2970550
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 628. Single-Shot Refinement Neural Network for Object Detection
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：架构论文
- 引用数：1300
- 年份：2018
- 作者：Shifeng Zhang, Longyin Wen, Xiao Bian et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00442
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 629. Complete discrete 2-D Gabor transforms by neural networks for image analysis and compression
- 类型：计算机视觉 / 语音/音频
- 标签：视觉, Speech
- 重要性：架构论文
- 引用数：1299
- 年份：1988
- 作者：J.G. Daugman
- 来源：IEEE Transactions on Acoustics, Speech, and Signal Processing
- 链接：https://doi.org/10.1109/29.1644
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 630. The impact of surface states on the DC and RF characteristics of AlGaN/GaN HFETs
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1296
- 年份：2001
- 作者：R. Vetury, N.Q. Zhang, S. Keller et al.
- 来源：IEEE Transactions on Electron Devices
- 链接：https://doi.org/10.1109/16.906451
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 631. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
- 类型：计算机视觉
- 标签：视觉, RNN
- 重要性：奠基/方法论文
- 引用数：1286
- 年份：2016
- 作者：Xuezhe Ma, Eduard Hovy
- 来源：Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 链接：https://doi.org/10.18653/v1/p16-1101
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 632. Deep Learning Applications in Medical Image Analysis
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：1279
- 年份：2018
- 作者：Justin Ker, Lipo Wang, Jai Rao et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2017.2788044
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 633. Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks
- 类型：医疗/生命科学AI
- 标签：Medical AI, RNN
- 重要性：架构论文
- 引用数：1279
- 年份：2018
- 作者：Marwin H. S. Segler, Thierry Kogej, Christian Tyrchan et al.
- 来源：ACS Central Science
- 链接：https://doi.org/10.1021/acscentsci.7b00512
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 634. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1276
- 年份：2018
- 作者：Sergey Levine, Peter Pastor, Alex Krizhevsky et al.
- 来源：The International Journal of Robotics Research
- 链接：https://doi.org/10.1177/0278364917710318
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 635. LSTM Fully Convolutional Networks for Time Series Classification
- 类型：计算机视觉
- 标签：视觉, CNN, RNN
- 重要性：奠基/方法论文
- 引用数：1272
- 年份：2018
- 作者：Fazle Karim, Somshubra Majumdar, Houshang Darabi et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2017.2779939
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 636. A deep convolutional neural network model to classify heartbeats
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1270
- 年份：2017
- 作者：U. Rajendra Acharya, Shu Lih Oh, Yuki Hagiwara et al.
- 来源：Computers in Biology and Medicine
- 链接：https://doi.org/10.1016/j.compbiomed.2017.08.022
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 637. A Bayesian neural network method for adverse drug reaction signal generation
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1270
- 年份：1998
- 作者：A. Bate, M. Lindquist, I. R. Edwards et al.
- 来源：European Journal of Clinical Pharmacology
- 链接：https://doi.org/10.1007/s002280050466
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 638. The 2018 GaN power electronics roadmap
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1269
- 年份：2018
- 作者：H Amano, Y Baines, E Beam et al.
- 来源：Journal of Physics D: Applied Physics
- 链接：https://doi.org/10.1088/1361-6463/aaaf9d
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 639. TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：1266
- 年份：2024
- 作者：Jieneng Chen, Jieru Mei, Xianhang Li et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2024.103280
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 640. Heterogeneous Graph Neural Network
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1265
- 年份：2019
- 作者：Chuxu Zhang, Dongjin Song, Chao Huang et al.
- 来源：Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp;amp; Data Mining
- 链接：https://doi.org/10.1145/3292500.3330961
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 641. Artificial Neural Networks in Hydrology. I: Preliminary Concepts
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1262
- 年份：2000
- 作者：—
- 来源：Journal of Hydrologic Engineering
- 链接：https://doi.org/10.1061/(asce)1084-0699(2000)5:2(115)
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 642. Target Classification Using the Deep Convolutional Networks for SAR Images
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1261
- 年份：2016
- 作者：Sizhe Chen, Haipeng Wang, Feng Xu et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2016.2551720
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 643. Neural networks for classification: a survey
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1261
- 年份：2000
- 作者：G.P. Zhang
- 来源：IEEE Transactions on Systems, Man and Cybernetics, Part C (Applications and Reviews)
- 链接：https://doi.org/10.1109/5326.897072
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 644. Nearly optimal control laws for nonlinear systems with saturating actuators using a neural network HJB approach
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1258
- 年份：2005
- 作者：Murad Abu-Khalaf, Frank L. Lewis
- 来源：Automatica
- 链接：https://doi.org/10.1016/j.automatica.2004.11.034
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 645. A comparative study of fine-tuning deep learning models for plant disease identification
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1257
- 年份：2019
- 作者：Edna Chebet Too, Li Yujian, Sam Njuki et al.
- 来源：Computers and Electronics in Agriculture
- 链接：https://doi.org/10.1016/j.compag.2018.03.032
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 646. D-NeRF: Neural Radiance Fields for Dynamic Scenes
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：1255
- 年份：2021
- 作者：Albert Pumarola, Enric Corona, Gerard Pons-Moll et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.01018
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 647. Deep learning for sentiment analysis: A survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1254
- 年份：2018
- 作者：Lei Zhang, Shuai Wang, Bing Liu
- 来源：WIREs Data Mining and Knowledge Discovery
- 链接：https://doi.org/10.1002/widm.1253
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 648. Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1254
- 年份：2018
- 作者：Pranav Rajpurkar, Jeremy Irvin, Robyn L. Ball et al.
- 来源：PLOS Medicine
- 链接：https://doi.org/10.1371/journal.pmed.1002686
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 649. Modeling of strength of high-performance concrete using artificial neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1252
- 年份：1998
- 作者：I.-C. Yeh
- 来源：Cement and Concrete Research
- 链接：https://doi.org/10.1016/s0008-8846(98)00165-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 650. Learning Traffic as Images: A Deep Convolutional Neural Network for Large-Scale Transportation Network Speed Prediction
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1249
- 年份：2017
- 作者：Xiaolei Ma, Zhuang Dai, Zhengbing He et al.
- 来源：Sensors
- 链接：https://doi.org/10.3390/s17040818
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 651. Heterogeneous Graph Transformer
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：1246
- 年份：2020
- 作者：Ziniu Hu, Yuxiao Dong, Kuansan Wang et al.
- 来源：Proceedings of The Web Conference 2020
- 链接：https://doi.org/10.1145/3366423.3380027
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 652. Supervised Speech Separation Based on Deep Learning: An Overview
- 类型：语音/音频
- 标签：Speech
- 重要性：综述论文
- 引用数：1243
- 年份：2018
- 作者：DeLiang Wang, Jitong Chen
- 来源：IEEE/ACM Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/taslp.2018.2842159
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 653. A survey of uncertainty in deep neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1242
- 年份：2023
- 作者：Jakob Gawlikowski, Cedrique Rovile Njieutcheu Tassi, Mohsin Ali et al.
- 来源：Artificial Intelligence Review
- 链接：https://doi.org/10.1007/s10462-023-10562-9
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 654. Deep learning in spiking neural networks
- 类型：深度学习相关
- 标签：Segmentation
- 重要性：架构论文
- 引用数：1241
- 年份：2019
- 作者：Amirhossein Tavanaei, Masoud Ghodrati, Saeed Reza Kheradpisheh et al.
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2018.12.002
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 655. Financial time series forecasting with deep learning : A systematic literature review: 2005–2019
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：1239
- 年份：2020
- 作者：Omer Berat Sezer, Mehmet Ugur Gudelek, Ahmet Murat Ozbayoglu
- 来源：Applied Soft Computing
- 链接：https://doi.org/10.1016/j.asoc.2020.106181
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 656. Multi-focus image fusion with a deep convolutional neural network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1239
- 年份：2017
- 作者：Yu Liu, Xun Chen, Hu Peng et al.
- 来源：Information Fusion
- 链接：https://doi.org/10.1016/j.inffus.2016.12.001
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 657. Deep learning for universal linear embeddings of nonlinear dynamics
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1235
- 年份：2018
- 作者：Bethany Lusch, J. Nathan Kutz, Steven L. Brunton
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/s41467-018-07210-0
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 658. Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1234
- 年份：2021
- 作者：Vishal Monga, Yuelong Li, Yonina C. Eldar
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/msp.2020.3016905
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 659. Modeling polypharmacy side effects with graph convolutional networks
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：架构论文
- 引用数：1230
- 年份：2018
- 作者：Marinka Zitnik, Monica Agrawal, Jure Leskovec
- 来源：Bioinformatics
- 链接：https://doi.org/10.1093/bioinformatics/bty294
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 660. Automatic detection of coronavirus disease (COVID-19) using X-ray images and deep convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1228
- 年份：2021
- 作者：Ali Narin, Ceren Kaya, Ziynet Pamuk
- 来源：Pattern Analysis and Applications
- 链接：https://doi.org/10.1007/s10044-021-00984-y
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 661. Deep Learning for IoT Big Data and Streaming Analytics: A Survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1225
- 年份：2018
- 作者：Mehdi Mohammadi, Ala Al-Fuqaha, Sameh Sorour et al.
- 来源：IEEE Communications Surveys &amp;amp; Tutorials
- 链接：https://doi.org/10.1109/comst.2018.2844341
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 662. Effects of eddy currents in transformer windings
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：1225
- 年份：1966
- 作者：P.L. Dowell
- 来源：Proceedings of the Institution of Electrical Engineers
- 链接：https://doi.org/10.1049/piee.1966.0236
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 663. Improved Adam Optimizer for Deep Neural Networks
- 类型：深度学习相关
- 标签：Optimization/Training
- 重要性：奠基/方法论文
- 引用数：1222
- 年份：2018
- 作者：Zijun Zhang
- 来源：2018 IEEE/ACM 26th International Symposium on Quality of Service (IWQoS)
- 链接：https://doi.org/10.1109/iwqos.2018.8624183
- 概述：提出或分析 Adam/自适应梯度优化方法，用于稳定高效训练深度网络。

## 664. Data augmentation for improving deep learning in image classification problem
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1222
- 年份：2018
- 作者：Agnieszka Mikolajczyk, Michal Grochowski
- 来源：2018 International Interdisciplinary PhD Workshop (IIPhDW)
- 链接：https://doi.org/10.1109/iiphdw.2018.8388338
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 665. Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1221
- 年份：2017
- 作者：Ying Li, Haokui Zhang, Qiang Shen
- 来源：Remote Sensing
- 链接：https://doi.org/10.3390/rs9010067
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 666. A Neural Attention Model for Abstractive Sentence Summarization
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：架构论文
- 引用数：1221
- 年份：2015
- 作者：Alexander M. Rush, Sumit Chopra, Jason Weston
- 来源：Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing
- 链接：https://doi.org/10.18653/v1/d15-1044
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 667. Quantum-chemical insights from deep tensor neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1220
- 年份：2017
- 作者：Kristof T. Schütt, Farhad Arbabzadah, Stefan Chmiela et al.
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/ncomms13890
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 668. Theory of the backpropagation neural network
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1219
- 年份：1989
- 作者：Hecht-Nielsen
- 来源：International Joint Conference on Neural Networks
- 链接：https://doi.org/10.1109/ijcnn.1989.118638
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 669. Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1219
- 年份：2020
- 作者：Haofan Wang, Zifan Wang, Mengnan Du et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- 链接：https://doi.org/10.1109/cvprw50498.2020.00020
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 670. Medical Image Analysis using Convolutional Neural Networks: A Review
- 类型：计算机视觉 / 医疗/生命科学AI / 系统/框架
- 标签：视觉, Medical AI, Framework/System, CNN
- 重要性：综述论文
- 引用数：1217
- 年份：2018
- 作者：Syed Muhammad Anwar, Muhammad Majid, Adnan Qayyum et al.
- 来源：Journal of Medical Systems
- 链接：https://doi.org/10.1007/s10916-018-1088-1
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 671. Artificial Neural Network Modeling of the Rainfall‐Runoff Process
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1217
- 年份：1995
- 作者：Kuo‐lin Hsu, Hoshin Vijai Gupta, Soroosh Sorooshian
- 来源：Water Resources Research
- 链接：https://doi.org/10.1029/95wr01955
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 672. Deep learning-enabled medical computer vision
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：1215
- 年份：2021
- 作者：Andre Esteva, Katherine Chou, Serena Yeung et al.
- 来源：npj Digital Medicine
- 链接：https://doi.org/10.1038/s41746-020-00376-2
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 673. Long Short-Term Memory Recurrent Neural Network for Remaining Useful Life Prediction of Lithium-Ion Batteries
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：1212
- 年份：2018
- 作者：Yongzhi Zhang, Rui Xiong, Hongwen He et al.
- 来源：IEEE Transactions on Vehicular Technology
- 链接：https://doi.org/10.1109/tvt.2018.2805189
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 674. Acoustic Modeling Using Deep Belief Networks
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：1211
- 年份：2012
- 作者：Abdel-rahman Mohamed, George E. Dahl, Geoffrey Hinton
- 来源：IEEE Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/tasl.2011.2109382
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 675. A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1211
- 年份：2021
- 作者：Ehsan Haghighat, Maziar Raissi, Adrian Moure et al.
- 来源：Computer Methods in Applied Mechanics and Engineering
- 链接：https://doi.org/10.1016/j.cma.2021.113741
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 676. FcaNet: Frequency Channel Attention Networks
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1211
- 年份：2021
- 作者：Zequn Qin, Pengyi Zhang, Fei Wu et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00082
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 677. Neural Network-Based Adaptive Dynamic Surface Control for a Class of Uncertain Nonlinear Systems in Strict-Feedback Form
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System
- 重要性：系统/框架
- 引用数：1210
- 年份：2005
- 作者：D. Wang, J. Huang
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/tnn.2004.839354
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 678. Increase in the extraction efficiency of GaN-based light-emitting diodes via surface roughening
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1210
- 年份：2004
- 作者：T. Fujii, Y. Gao, R. Sharma et al.
- 来源：Applied Physics Letters
- 链接：https://doi.org/10.1063/1.1645992
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 679. Real-time vibration-based structural damage detection using one-dimensional convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1209
- 年份：2017
- 作者：Osama Abdeljaber, Onur Avci, Serkan Kiranyaz et al.
- 来源：Journal of Sound and Vibration
- 链接：https://doi.org/10.1016/j.jsv.2016.10.043
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 680. A Survey of Deep Learning-Based Object Detection
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：综述论文
- 引用数：1205
- 年份：2019
- 作者：Licheng Jiao, Fan Zhang, Fang Liu et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2019.2939201
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 681. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1204
- 年份：2021
- 作者：Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00713
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 682. GaN Growth Using GaN Buffer Layer
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1203
- 年份：1991
- 作者：Shuji Nakamura Shuji Nakamura
- 来源：Japanese Journal of Applied Physics
- 链接：https://doi.org/10.1143/jjap.30.l1705
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 683. Adaptive Neural Network Control of an Uncertain Robot With Full-State Constraints
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1202
- 年份：2016
- 作者：Wei He, Yuhao Chen, Zhao Yin
- 来源：IEEE Transactions on Cybernetics
- 链接：https://doi.org/10.1109/tcyb.2015.2411285
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 684. Harnessing protein folding neural networks for peptide–protein docking
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1200
- 年份：2022
- 作者：Tomer Tsaban, Julia K. Varga, Orly Avraham et al.
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/s41467-021-27838-9
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 685. Optimal unsupervised learning in a single-layer linear feedforward neural network
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1198
- 年份：1989
- 作者：Terence D. Sanger
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/0893-6080(89)90044-0
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 686. Review of Commercial GaN Power Devices and GaN-Based Converter Design Challenges
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：综述论文
- 引用数：1197
- 年份：2016
- 作者：Edward A. Jones, Fei Fred Wang, Daniel Costinett
- 来源：IEEE Journal of Emerging and Selected Topics in Power Electronics
- 链接：https://doi.org/10.1109/jestpe.2016.2582685
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 687. Using LSTM and GRU neural network methods for traffic flow prediction
- 类型：深度学习相关
- 标签：RNN
- 重要性：奠基/方法论文
- 引用数：1193
- 年份：2016
- 作者：Rui Fu, Zuo Zhang, Li Li
- 来源：2016 31st Youth Academic Annual Conference of Chinese Association of Automation (YAC)
- 链接：https://doi.org/10.1109/yac.2016.7804912
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 688. A Survey on Deep Learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1193
- 年份：2019
- 作者：Samira Pouyanfar, Saad Sadiq, Yilin Yan et al.
- 来源：ACM Computing Surveys
- 链接：https://doi.org/10.1145/3234150
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 689. Automated Identification of Diabetic Retinopathy Using Deep Learning
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：1193
- 年份：2017
- 作者：Rishab Gargeya, Theodore Leng
- 来源：Ophthalmology
- 链接：https://doi.org/10.1016/j.ophtha.2017.02.008
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 690. THE PREPARATION AND PROPERTIES OF VAPOR-DEPOSITED SINGLE-CRYSTAL-LINE GaN
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1193
- 年份：1969
- 作者：H. P. Maruska, J. J. Tietjen
- 来源：Applied Physics Letters
- 链接：https://doi.org/10.1063/1.1652845
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 691. Explainable artificial intelligence (XAI) in deep learning-based medical image analysis
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：1191
- 年份：2022
- 作者：Bas H.M. van der Velden, Hugo J. Kuijf, Kenneth G.A. Gilhuijs et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2022.102470
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 692. Origin of efficiency droop in GaN-based light-emitting diodes
- 类型：NLP/语言模型 / 生成模型
- 标签：Language Model, Generative AI, Attention/Transformer, GAN
- 重要性：应用/方法论文
- 引用数：1188
- 年份：2007
- 作者：Min-Ho Kim, Martin F. Schubert, Qi Dai et al.
- 来源：Applied Physics Letters
- 链接：https://doi.org/10.1063/1.2800290
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 693. Recurrent Convolutional Neural Networks for Text Classification
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, CNN, RNN
- 重要性：架构论文
- 引用数：1186
- 年份：2015
- 作者：Siwei Lai, Liheng Xu, Kang Liu et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v29i1.9513
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 694. Road crack detection using deep convolutional neural network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1185
- 年份：2016
- 作者：Lei Zhang, Fan Yang, Yimin Daniel Zhang et al.
- 来源：2016 IEEE International Conference on Image Processing (ICIP)
- 链接：https://doi.org/10.1109/icip.2016.7533052
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 695. Recurrent Neural Networks for Time Series Forecasting: Current status and future directions
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1183
- 年份：2021
- 作者：Hansika Hewamalage, Christoph Bergmeir, Kasun Bandara
- 来源：International Journal of Forecasting
- 链接：https://doi.org/10.1016/j.ijforecast.2020.06.008
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 696. Approximation theory of the MLP model in neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1183
- 年份：1999
- 作者：Allan Pinkus
- 来源：Acta Numerica
- 链接：https://doi.org/10.1017/s0962492900002919
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 697. Artificial Neural Networks in Hydrology. II: Hydrologic Applications
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1179
- 年份：2000
- 作者：—
- 来源：Journal of Hydrologic Engineering
- 链接：https://doi.org/10.1061/(asce)1084-0699(2000)5:2(124)
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 698. Privacy-Preserving Deep Learning via Additively Homomorphic Encryption
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1175
- 年份：2018
- 作者：Le Trieu Phong, Yoshinori Aono, Takuya Hayashi et al.
- 来源：IEEE Transactions on Information Forensics and Security
- 链接：https://doi.org/10.1109/tifs.2017.2787987
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 699. Deep Convolutional Transfer Learning Network: A New Method for Intelligent Fault Diagnosis of Machines With Unlabeled Data
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1175
- 年份：2019
- 作者：Liang Guo, Yaguo Lei, Saibo Xing et al.
- 来源：IEEE Transactions on Industrial Electronics
- 链接：https://doi.org/10.1109/tie.2018.2877090
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 700. A Survey on Deep Learning for Named Entity Recognition
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1175
- 年份：2022
- 作者：Jing Li, Aixin Sun, Jianglei Han et al.
- 来源：IEEE Transactions on Knowledge and Data Engineering
- 链接：https://doi.org/10.1109/tkde.2020.2981314
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 701. Deep learning for cellular image analysis
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1172
- 年份：2019
- 作者：Erick Moen, Dylan Bannon, Takamasa Kudo et al.
- 来源：Nature Methods
- 链接：https://doi.org/10.1038/s41592-019-0403-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 702. Physics-Informed Neural Networks for Heat Transfer Problems
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1172
- 年份：2021
- 作者：Shengze Cai, Zhicheng Wang, Sifan Wang et al.
- 来源：Journal of Heat Transfer
- 链接：https://doi.org/10.1115/1.4050542
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 703. Brain tumor classification using deep CNN features via transfer learning
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1171
- 年份：2019
- 作者：S. Deepak, P.M. Ameer
- 来源：Computers in Biology and Medicine
- 链接：https://doi.org/10.1016/j.compbiomed.2019.103345
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 704. Remote Sensing Image Change Detection With Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1171
- 年份：2022
- 作者：Hao Chen, Zipeng Qi, Zhenwei Shi
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2021.3095166
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 705. Bag of Tricks for Image Classification with Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1169
- 年份：2019
- 作者：Tong He, Zhi Zhang, Hang Zhang et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00065
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 706. Deep reinforcement learning for de novo drug design
- 类型：强化学习
- 标签：RL
- 重要性：训练/推理方法
- 引用数：1167
- 年份：2018
- 作者：Mariya Popova, Olexandr Isayev, Alexander Tropsha
- 来源：Science Advances
- 链接：https://doi.org/10.1126/sciadv.aap7885
- 概述：研究深度强化学习，将神经网络函数逼近用于决策、控制或游戏任务。

## 707. Convolutional Neural Network Based Fault Detection for Rotating Machinery
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1166
- 年份：2016
- 作者：Olivier Janssens, Viktor Slavkovikj, Bram Vervisch et al.
- 来源：Journal of Sound and Vibration
- 链接：https://doi.org/10.1016/j.jsv.2016.05.027
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 708. Harmonious Attention Network for Person Re-identification
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1163
- 年份：2018
- 作者：Wei Li, Xiatian Zhu, Shaogang Gong
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00243
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 709. CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1161
- 年份：2022
- 作者：Xiaoyi Dong, Jianmin Bao, Dongdong Chen et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01181
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 710. GraphDTA: predicting drug–target binding affinity with graph neural networks
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1161
- 年份：2021
- 作者：Thin Nguyen, Hang Le, Thomas P Quinn et al.
- 来源：Bioinformatics
- 链接：https://doi.org/10.1093/bioinformatics/btaa921
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 711. YOLO-v1 to YOLO-v8, the Rise of YOLO and Its Complementary Nature toward Digital Manufacturing and Industrial Defect Detection
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：应用/方法论文
- 引用数：1158
- 年份：2023
- 作者：Muhammad Hussain
- 来源：Machines
- 链接：https://doi.org/10.3390/machines11070677
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 712. Collaborative Deep Learning for Recommender Systems
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1157
- 年份：2015
- 作者：Hao Wang, Naiyan Wang, Dit-Yan Yeung
- 来源：Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
- 链接：https://doi.org/10.1145/2783258.2783273
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 713. Spatial Neglect and Attention Networks
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：综述论文
- 引用数：1153
- 年份：2011
- 作者：Maurizio Corbetta, Gordon L. Shulman
- 来源：Annual Review of Neuroscience
- 链接：https://doi.org/10.1146/annurev-neuro-061010-113731
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 714. Lung Pattern Classification for Interstitial Lung Diseases Using a Deep Convolutional Neural Network
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：1151
- 年份：2016
- 作者：Marios Anthimopoulos, Stergios Christodoulidis, Lukas Ebner et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2535865
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 715. Restricted Boltzmann machines for collaborative filtering
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：1150
- 年份：2007
- 作者：Ruslan Salakhutdinov, Andriy Mnih, Geoffrey Hinton
- 来源：Proceedings of the 24th international conference on Machine learning
- 链接：https://doi.org/10.1145/1273496.1273596
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 716. A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
- 类型：深度学习相关
- 标签：Attention/Transformer, RNN
- 重要性：架构论文
- 引用数：1149
- 年份：2017
- 作者：Yao Qin, Dongjin Song, Haifeng Chen et al.
- 来源：Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence
- 链接：https://doi.org/10.24963/ijcai.2017/366
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 717. The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：应用/方法论文
- 引用数：1148
- 年份：2017
- 作者：Simon Jegou, Michal Drozdzal, David Vazquez et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- 链接：https://doi.org/10.1109/cvprw.2017.156
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 718. Graph neural network for traffic forecasting: A survey
- 类型：图学习 / 系统/框架
- 标签：Graph, Framework/System
- 重要性：综述论文
- 引用数：1147
- 年份：2022
- 作者：Weiwei Jiang, Jiayun Luo
- 来源：Expert Systems with Applications
- 链接：https://doi.org/10.1016/j.eswa.2022.117921
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 719. Spectral–Spatial Classification of Hyperspectral Data Based on Deep Belief Network
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1143
- 年份：2015
- 作者：Yushi Chen, Xing Zhao, Xiuping Jia
- 来源：IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- 链接：https://doi.org/10.1109/jstars.2015.2388577
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 720. A recurrent neural network based health indicator for remaining useful life prediction of bearings
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1142
- 年份：2017
- 作者：Liang Guo, Naipeng Li, Feng Jia et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2017.02.045
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 721. Learning and development in neural networks: the importance of starting small
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1139
- 年份：1993
- 作者：Jeffrey L. Elman
- 来源：Cognition
- 链接：https://doi.org/10.1016/0010-0277(93)90058-4
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 722. Fine-Tuning CNN Image Retrieval with No Human Annotation
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1138
- 年份：2019
- 作者：Filip Radenovic, Giorgos Tolias, Ondrej Chum
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2018.2846566
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 723. Chaotic neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1137
- 年份：1990
- 作者：K. Aihara, T. Takabe, M. Toyoda
- 来源：Physics Letters A
- 链接：https://doi.org/10.1016/0375-9601(90)90136-c
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 724. Combining evolutionary information and neural networks to predict protein secondary structure
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1137
- 年份：1994
- 作者：Burkhard Rost, Chris Sander
- 来源：Proteins: Structure, Function, and Bioinformatics
- 链接：https://doi.org/10.1002/prot.340190108
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 725. Physics-informed neural networks for high-speed flows
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1136
- 年份：2020
- 作者：Zhiping Mao, Ameya D. Jagtap, George Em Karniadakis
- 来源：Computer Methods in Applied Mechanics and Engineering
- 链接：https://doi.org/10.1016/j.cma.2019.112789
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 726. 11 TOPS photonic convolutional accelerator for optical neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1135
- 年份：2021
- 作者：Xingyuan Xu, Mengxi Tan, Bill Corcoran et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-020-03063-0
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 727. Protein backbone and sidechain torsion angles predicted from NMR chemical shifts using artificial neural networks
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1133
- 年份：2013
- 作者：Yang Shen, Ad Bax
- 来源：Journal of Biomolecular NMR
- 链接：https://doi.org/10.1007/s10858-013-9741-y
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 728. Convolutional, Long Short-Term Memory, fully connected Deep Neural Networks
- 类型：NLP/语言模型
- 标签：RNN, LSTM, Sequence Modeling
- 重要性：奠基/方法论文
- 引用数：1133
- 年份：2015
- 作者：Tara N. Sainath, Oriol Vinyals, Andrew Senior et al.
- 来源：2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- 链接：https://doi.org/10.1109/icassp.2015.7178838
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 729. Learning Spatio-Temporal Transformer for Visual Tracking
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1132
- 年份：2021
- 作者：Bin Yan, Houwen Peng, Jianlong Fu et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.01028
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 730. Fuzzy logic, neural networks, and soft computing
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1132
- 年份：1994
- 作者：Lotfi A. Zadeh
- 来源：Communications of the ACM
- 链接：https://doi.org/10.1145/175247.175255
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 731. Graph Neural Network-Based Anomaly Detection in Multivariate Time Series
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：1129
- 年份：2021
- 作者：Ailin Deng, Bryan Hooi
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v35i5.16523
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 732. Anomaly Detection with Robust Deep Autoencoders
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：1129
- 年份：2017
- 作者：Chong Zhou, Randy C. Paffenroth
- 来源：Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
- 链接：https://doi.org/10.1145/3097983.3098052
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 733. Convolutional neural networks at constrained time cost
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1128
- 年份：2015
- 作者：Kaiming He, Jian Sun
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7299173
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 734. Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
- 类型：系统/框架
- 标签：Framework/System, Unsupervised Representation
- 重要性：系统/框架
- 引用数：1124
- 年份：2018
- 作者：Yisroel Mirsky, Tomer Doitshman, Yuval Elovici et al.
- 来源：Proceedings 2018 Network and Distributed System Security Symposium
- 链接：https://doi.org/10.14722/ndss.2018.23204
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 735. Electric load forecasting using an artificial neural network
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1122
- 年份：1991
- 作者：D.C. Park, M.A. El-Sharkawi, R.J. Marks et al.
- 来源：IEEE Transactions on Power Systems
- 链接：https://doi.org/10.1109/59.76685
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 736. D2-Net: A Trainable CNN for Joint Description and Detection of Local Features
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1116
- 年份：2019
- 作者：Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00828
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 737. A review of convolutional neural networks in computer vision
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：综述论文
- 引用数：1112
- 年份：2024
- 作者：Xia Zhao, Limin Wang, Yufei Zhang et al.
- 来源：Artificial Intelligence Review
- 链接：https://doi.org/10.1007/s10462-024-10721-6
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 738. Ultrafast machine vision with 2D material neural network image sensors
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1110
- 年份：2020
- 作者：Lukas Mennel, Joanna Symonowicz, Stefan Wachter et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/s41586-020-2038-x
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 739. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1110
- 年份：2018
- 作者：Sebastian Bosse, Dominique Maniry, Klaus-Robert Muller et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2017.2760518
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 740. Deep Models Under the GAN
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1109
- 年份：2017
- 作者：Briland Hitaj, Giuseppe Ateniese, Fernando Perez-Cruz
- 来源：Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security
- 链接：https://doi.org/10.1145/3133956.3134012
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 741. A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：1107
- 年份：2018
- 作者：Jo Schlemper, Jose Caballero, Joseph V. Hajnal et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2017.2760978
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 742. Research on Attention Networks as a Model for the Integration of Psychological Science
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：综述论文
- 引用数：1106
- 年份：2007
- 作者：Michael I. Posner, Mary K. Rothbart
- 来源：Annual Review of Psychology
- 链接：https://doi.org/10.1146/annurev.psych.58.110405.085516
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 743. TransReID: Transformer-based Object Re-Identification
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1105
- 年份：2021
- 作者：Shuting He, Hao Luo, Pichao Wang et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.01474
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 744. Learning to compare image patches via convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1102
- 年份：2015
- 作者：Sergey Zagoruyko, Nikos Komodakis
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7299064
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 745. Pansharpening by Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1101
- 年份：2016
- 作者：Giuseppe Masi, Davide Cozzolino, Luisa Verdoliva et al.
- 来源：Remote Sensing
- 链接：https://doi.org/10.3390/rs8070594
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 746. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1096
- 年份：2020
- 作者：Ameya D. Jagtap, Ehsan Kharazmi, George Em Karniadakis
- 来源：Computer Methods in Applied Mechanics and Engineering
- 链接：https://doi.org/10.1016/j.cma.2020.113028
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 747. BadNets: Evaluating Backdooring Attacks on Deep Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1091
- 年份：2019
- 作者：Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2019.2909068
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 748. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：1090
- 年份：2020
- 作者：Qiuqiang Kong, Yin Cao, Turab Iqbal et al.
- 来源：IEEE/ACM Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/taslp.2020.3030497
- 概述：研究深度语音/音频建模，服务于识别、分离或表征学习任务。

## 749. Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1090
- 年份：2018
- 作者：Dianjing Liu, Yixuan Tan, Erfan Khoram et al.
- 来源：ACS Photonics
- 链接：https://doi.org/10.1021/acsphotonics.7b01377
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 750. Variational Autoencoders for Collaborative Filtering
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：1090
- 年份：2018
- 作者：Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman et al.
- 来源：Proceedings of the 2018 World Wide Web Conference on World Wide Web - WWW '18
- 链接：https://doi.org/10.1145/3178876.3186150
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 751. On the Continuity of Rotation Representations in Neural Networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1088
- 年份：2019
- 作者：Yi Zhou, Connelly Barnes, Jingwan Lu et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00589
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 752. Learned Image Compression With Discretized Gaussian Mixture Likelihoods and Attention Modules
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1086
- 年份：2020
- 作者：Zhengxue Cheng, Heming Sun, Masaru Takeuchi et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.00796
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 753. Autonomous concrete crack detection using deep fully convolutional neural network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1085
- 年份：2019
- 作者：Cao Vu Dung, Le Duc Anh
- 来源：Automation in Construction
- 链接：https://doi.org/10.1016/j.autcon.2018.11.028
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 754. Recurrent neural networks and robust time series prediction
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1085
- 年份：1994
- 作者：J.T. Connor, R.D. Martin, L.E. Atlas
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.279188
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 755. Training cost-sensitive neural networks with methods addressing the class imbalance problem
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1085
- 年份：2006
- 作者：Zhi-Hua Zhou, Xu-Ying Liu
- 来源：IEEE Transactions on Knowledge and Data Engineering
- 链接：https://doi.org/10.1109/tkde.2006.17
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 756. Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1083
- 年份：2019
- 作者：Bolun Wang, Yuanshun Yao, Shawn Shan et al.
- 来源：2019 IEEE Symposium on Security and Privacy (SP)
- 链接：https://doi.org/10.1109/sp.2019.00031
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 757. Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, CNN, RNN
- 重要性：架构论文
- 引用数：1082
- 年份：2017
- 作者：Jianlong Fu, Heliang Zheng, Tao Mei
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.476
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 758. Transferring Deep Convolutional Neural Networks for the Scene Classification of High-Resolution Remote Sensing Imagery
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1076
- 年份：2015
- 作者：Fan Hu, Gui-Song Xia, Jingwen Hu et al.
- 来源：Remote Sensing
- 链接：https://doi.org/10.3390/rs71114680
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 759. Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：1074
- 年份：2016
- 作者：Korsuk Sirinukunwattana, Shan E Ahmed Raza, Yee-Wah Tsang et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2525803
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 760. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1074
- 年份：1995
- 作者：Tianping Chen, Hong Chen
- 来源：IEEE Transactions on Neural Networks
- 链接：https://doi.org/10.1109/72.392253
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 761. Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：1072
- 年份：2017
- 作者：Konstantinos Bousmalis, Nathan Silberman, David Dohan et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.18
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 762. Pulmonary Nodule Detection in CT Images: False Positive Reduction Using Multi-View Convolutional Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：1072
- 年份：2016
- 作者：Arnaud Arindra Adiyoso Setio, Francesco Ciompi, Geert Litjens et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2536809
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 763. Curiosity-Driven Exploration by Self-Supervised Prediction
- 类型：计算机视觉
- 标签：视觉, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：1071
- 年份：2017
- 作者：Deepak Pathak, Pulkit Agrawal, Alexei A. Efros et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
- 链接：https://doi.org/10.1109/cvprw.2017.70
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 764. BiFormer: Vision Transformer with Bi-Level Routing Attention
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1070
- 年份：2023
- 作者：Lei Zhu, Xinjiang Wang, Zhanghan Ke et al.
- 来源：2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52729.2023.00995
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 765. Gallium vacancies and the yellow luminescence in GaN
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1070
- 年份：1996
- 作者：Jörg Neugebauer, Chris G. Van de Walle
- 来源：Applied Physics Letters
- 链接：https://doi.org/10.1063/1.117767
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 766. BioGPT: generative pre-trained transformer for biomedical text generation and mining
- 类型：NLP/语言模型 / 生成模型 / 医疗/生命科学AI
- 标签：Language Model, Generative AI, Medical AI, Attention/Transformer
- 重要性：架构论文
- 引用数：1069
- 年份：2022
- 作者：Renqian Luo, Liai Sun, Yingce Xia et al.
- 来源：Briefings in Bioinformatics
- 链接：https://doi.org/10.1093/bib/bbac409
- 概述：围绕Transformer预训练语言模型，推动NLP迁移学习、上下文学习和大模型范式。

## 767. A Regression Approach to Speech Enhancement Based on Deep Neural Networks
- 类型：语音/音频
- 标签：Speech
- 重要性：架构论文
- 引用数：1068
- 年份：2015
- 作者：Yong Xu, Jun Du, Li-Rong Dai et al.
- 来源：IEEE/ACM Transactions on Audio, Speech, and Language Processing
- 链接：https://doi.org/10.1109/taslp.2014.2364452
- 概述：研究深度语音/音频建模，服务于识别、分离或表征学习任务。

## 768. A Transformer-Based Siamese Network for Change Detection
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1067
- 年份：2022
- 作者：Wele Gedara Chaminda Bandara, Vishal M. Patel
- 来源：IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium
- 链接：https://doi.org/10.1109/igarss46834.2022.9883686
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 769. CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest x-ray images
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：1067
- 年份：2020
- 作者：Asif Iqbal Khan, Junaid Latief Shah, Mohammad Mudasir Bhat
- 来源：Computer Methods and Programs in Biomedicine
- 链接：https://doi.org/10.1016/j.cmpb.2020.105581
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 770. Human activity recognition with smartphone sensors using deep learning neural networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1065
- 年份：2016
- 作者：Charissa Ann Ronao, Sung-Bae Cho
- 来源：Expert Systems with Applications
- 链接：https://doi.org/10.1016/j.eswa.2016.04.032
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 771. Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction
- 类型：深度学习相关
- 标签：Unsupervised Representation
- 重要性：应用/方法论文
- 引用数：1064
- 年份：2014
- 作者：Mayu Sakurada, Takehisa Yairi
- 来源：Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis
- 链接：https://doi.org/10.1145/2689746.2689747
- 概述：研究无监督表征学习/自编码器或玻尔兹曼机，为深层模型预训练与特征学习奠基。

## 772. Efficient Geometry-aware 3D Generative Adversarial Networks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：架构论文
- 引用数：1063
- 年份：2022
- 作者：Eric R. Chan, Connor Z. Lin, Matthew A. Chan et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.01565
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 773. Going Deeper in Spiking Neural Networks: VGG and Residual Architectures
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer, Residual/Skip Connection, CNN
- 重要性：奠基/方法论文
- 引用数：1062
- 年份：2019
- 作者：Abhronil Sengupta, Yuting Ye, Robert Wang et al.
- 来源：Frontiers in Neuroscience
- 链接：https://doi.org/10.3389/fnins.2019.00095
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 774. MoDL: Model-Based Deep Learning Architecture for Inverse Problems
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：1059
- 年份：2019
- 作者：Hemant K. Aggarwal, Merry P. Mani, Mathews Jacob
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2018.2865356
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 775. Graph Neural Networks in Recommender Systems: A Survey
- 类型：图学习 / 系统/框架
- 标签：Graph, Framework/System
- 重要性：综述论文
- 引用数：1058
- 年份：2023
- 作者：Shiwen Wu, Fei Sun, Wentao Zhang et al.
- 来源：ACM Computing Surveys
- 链接：https://doi.org/10.1145/3535101
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 776. NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1055
- 年份：2021
- 作者：Xiaowei Jin, Shengze Cai, Hui Li et al.
- 来源：Journal of Computational Physics
- 链接：https://doi.org/10.1016/j.jcp.2020.109951
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 777. Deep Architecture for Traffic Flow Prediction: Deep Belief Networks With Multitask Learning
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1053
- 年份：2014
- 作者：Wenhao Huang, Guojie Song, Haikun Hong et al.
- 来源：IEEE Transactions on Intelligent Transportation Systems
- 链接：https://doi.org/10.1109/tits.2014.2311123
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 778. Meta-Learning in Neural Networks: A Survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1051
- 年份：2021
- 作者：Timothy M Hospedales, Antreas Antoniou, Paul Micaelli et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2021.3079209
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 779. Review of Solid-State Transformer Technologies and Their Application in Power Distribution Systems
- 类型：系统/框架
- 标签：Framework/System, Attention/Transformer
- 重要性：综述论文
- 引用数：1050
- 年份：2013
- 作者：Xu She, Alex Q. Huang, Rolando Burgos
- 来源：IEEE Journal of Emerging and Selected Topics in Power Electronics
- 链接：https://doi.org/10.1109/jestpe.2013.2277917
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 780. Network intrusion detection system: A systematic study of machine learning and deep learning approaches
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1048
- 年份：2021
- 作者：Zeeshan Ahmad, Adnan Shahid Khan, Cheah Wai Shiang et al.
- 来源：Transactions on Emerging Telecommunications Technologies
- 链接：https://doi.org/10.1002/ett.4150
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 781. Illuminating the “black box”: a randomization approach for understanding variable contributions in artificial neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1048
- 年份：2002
- 作者：Julian D Olden, Donald A Jackson
- 来源：Ecological Modelling
- 链接：https://doi.org/10.1016/s0304-3800(02)00064-9
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 782. Fully memristive neural networks for pattern classification with unsupervised learning
- 类型：深度学习相关
- 标签：RNN
- 重要性：架构论文
- 引用数：1047
- 年份：2018
- 作者：Zhongrui Wang, Saumil Joshi, Sergey Savel’ev et al.
- 来源：Nature Electronics
- 链接：https://doi.org/10.1038/s41928-018-0023-2
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 783. Review and comparison of methods to study the contribution of variables in artificial neural network models
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：1046
- 年份：2003
- 作者：Muriel Gevrey, Ioannis Dimopoulos, Sovan Lek
- 来源：Ecological Modelling
- 链接：https://doi.org/10.1016/s0304-3800(02)00257-0
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 784. Self-supervised Learning: Generative or Contrastive
- 类型：生成模型
- 标签：Generative AI, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：1045
- 年份：2021
- 作者：Xiao Liu, Fanjin Zhang, Zhenyu Hou et al.
- 来源：IEEE Transactions on Knowledge and Data Engineering
- 链接：https://doi.org/10.1109/tkde.2021.3090866
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 785. Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1044
- 年份：2016
- 作者：David R. Kelley, Jasper Snoek, John L. Rinn
- 来源：Genome Research
- 链接：https://doi.org/10.1101/gr.200535.115
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 786. B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1044
- 年份：2021
- 作者：Liu Yang, Xuhui Meng, George Em Karniadakis
- 来源：Journal of Computational Physics
- 链接：https://doi.org/10.1016/j.jcp.2020.109913
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 787. Bidirectional LSTM with attention mechanism and convolutional layer for text classification
- 类型：NLP/语言模型 / 计算机视觉
- 标签：Language Model, 视觉, Attention/Transformer, CNN, RNN
- 重要性：奠基/方法论文
- 引用数：1043
- 年份：2019
- 作者：Gang Liu, Jiabao Guo
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2019.01.078
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 788. A Transformer-based Framework for Multivariate Time Series Representation Learning
- 类型：系统/框架
- 标签：Framework/System, Attention/Transformer
- 重要性：系统/框架
- 引用数：1040
- 年份：2021
- 作者：George Zerveas, Srideepika Jayaraman, Dhaval Patel et al.
- 来源：Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp;amp; Data Mining
- 链接：https://doi.org/10.1145/3447548.3467401
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 789. Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：应用/方法论文
- 引用数：1039
- 年份：2016
- 作者：Fayao Liu, Chunhua Shen, Guosheng Lin et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2015.2505283
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 790. Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1038
- 年份：2018
- 作者：Yujie Wu, Lei Deng, Guoqi Li et al.
- 来源：Frontiers in Neuroscience
- 链接：https://doi.org/10.3389/fnins.2018.00331
- 概述：研究反向传播或梯度训练机制，是多层神经网络学习的核心基础。

## 791. Gapped sequence alignment using artificial neural networks: application to the MHC class I system
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1035
- 年份：2016
- 作者：Massimo Andreatta, Morten Nielsen
- 来源：Bioinformatics
- 链接：https://doi.org/10.1093/bioinformatics/btv639
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 792. EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization
- 类型：计算机视觉 / 语音/音频 / 系统/框架
- 标签：视觉, Speech, Framework/System, Attention/Transformer, CNN
- 重要性：系统/框架
- 引用数：1033
- 年份：2023
- 作者：Yonghao Song, Qingqing Zheng, Bingchuan Liu et al.
- 来源：IEEE Transactions on Neural Systems and Rehabilitation Engineering
- 链接：https://doi.org/10.1109/tnsre.2022.3230250
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 793. Rotate to Attend: Convolutional Triplet Attention Module
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, CNN
- 重要性：应用/方法论文
- 引用数：1031
- 年份：2021
- 作者：Diganta Misra, Trikay Nalamada, Ajay Uppili Arasanipalai et al.
- 来源：2021 IEEE Winter Conference on Applications of Computer Vision (WACV)
- 链接：https://doi.org/10.1109/wacv48630.2021.00318
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 794. UIU-Net: U-Net in U-Net for Infrared Small Object Detection
- 类型：计算机视觉
- 标签：视觉, Object Detection, Segmentation
- 重要性：架构论文
- 引用数：1031
- 年份：2023
- 作者：Xin Wu, Danfeng Hong, Jocelyn Chanussot
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2022.3228497
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 795. BranchyNet: Fast inference via early exiting from deep neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1030
- 年份：2016
- 作者：Surat Teerapittayanon, Bradley McDanel, H.T. Kung
- 来源：2016 23rd International Conference on Pattern Recognition (ICPR)
- 链接：https://doi.org/10.1109/icpr.2016.7900006
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 796. Deepfakes and beyond: A Survey of face manipulation and fake detection
- 类型：计算机视觉
- 标签：视觉
- 重要性：综述论文
- 引用数：1029
- 年份：2020
- 作者：Ruben Tolosana, Ruben Vera-Rodriguez, Julian Fierrez et al.
- 来源：Information Fusion
- 链接：https://doi.org/10.1016/j.inffus.2020.06.014
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 797. Regularization Theory and Neural Networks Architectures
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1027
- 年份：1995
- 作者：Federico Girosi, Michael Jones, Tomaso Poggio
- 来源：Neural Computation
- 链接：https://doi.org/10.1162/neco.1995.7.2.219
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 798. Generating Coherent Patterns of Activity from Chaotic Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：1026
- 年份：2009
- 作者：David Sussillo, L.F. Abbott
- 来源：Neuron
- 链接：https://doi.org/10.1016/j.neuron.2009.07.018
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 799. Nerfies: Deformable Neural Radiance Fields
- 类型：计算机视觉
- 标签：视觉
- 重要性：应用/方法论文
- 引用数：1026
- 年份：2021
- 作者：Keunhong Park, Utkarsh Sinha, Jonathan T. Barron et al.
- 来源：2021 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv48922.2021.00581
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 800. Activating More Pixels in Image Super-Resolution Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：1023
- 年份：2023
- 作者：Xiangyu Chen, Xintao Wang, Jiantao Zhou et al.
- 来源：2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52729.2023.02142
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 801. Thermal Annealing Effects on P-Type Mg-Doped GaN Films
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1022
- 年份：1992
- 作者：Shuji Nakamura, Takashi Mukai, Masayuki Senoh Masayuki Senoh et al.
- 来源：Japanese Journal of Applied Physics
- 链接：https://doi.org/10.1143/jjap.31.l139
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 802. CNN-Generated Images Are Surprisingly Easy to Spot… for Now
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1018
- 年份：2020
- 作者：Sheng-Yu Wang, Oliver Wang, Richard Zhang et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.00872
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 803. Convolutional Neural Networks for No-Reference Image Quality Assessment
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：1015
- 年份：2014
- 作者：Le Kang, Peng Ye, Yi Li et al.
- 来源：2014 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2014.224
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 804. Polarization effects, surface states, and the source of electrons in AlGaN/GaN heterostructure field effect transistors
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：1014
- 年份：2000
- 作者：J. P. Ibbetson, P. T. Fini, K. D. Ness et al.
- 来源：Applied Physics Letters
- 链接：https://doi.org/10.1063/1.126940
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 805. SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- 类型：生成模型 / 强化学习
- 标签：Generative AI, RL, Optimization/Training, GAN
- 重要性：奠基/方法论文
- 引用数：1011
- 年份：2017
- 作者：Lantao Yu, Weinan Zhang, Jun Wang et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v31i1.10804
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 806. Deep Neural Nets as a Method for Quantitative Structure–Activity Relationships
- 类型：NLP/语言模型
- 标签：Language Model, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：1010
- 年份：2015
- 作者：Junshui Ma, Robert P. Sheridan, Andy Liaw et al.
- 来源：Journal of Chemical Information and Modeling
- 链接：https://doi.org/10.1021/ci500747n
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 807. The CNN paradigm
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN
- 重要性：系统/框架
- 引用数：1007
- 年份：1993
- 作者：L.O. Chua, T. Roska
- 来源：IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications
- 链接：https://doi.org/10.1109/81.222795
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 808. Convolutional neural network: a review of models, methodologies and applications to object detection
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：综述论文
- 引用数：1005
- 年份：2020
- 作者：Anamika Dhillon, Gyanendra K. Verma
- 来源：Progress in Artificial Intelligence
- 链接：https://doi.org/10.1007/s13748-019-00203-0
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 809. Neural networks for nonlinear programming
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：1003
- 年份：1988
- 作者：M.P. Kennedy, L.O. Chua
- 来源：IEEE Transactions on Circuits and Systems
- 链接：https://doi.org/10.1109/31.1783
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 810. Progress in supervised neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：998
- 年份：1993
- 作者：D.R. Hush, B.G. Horne
- 来源：IEEE Signal Processing Magazine
- 链接：https://doi.org/10.1109/79.180705
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 811. Speech emotion recognition using deep 1D &amp;amp; 2D CNN LSTM networks
- 类型：计算机视觉 / 语音/音频 / 医疗/生命科学AI
- 标签：视觉, Speech, Medical AI, CNN, RNN
- 重要性：奠基/方法论文
- 引用数：996
- 年份：2019
- 作者：Jianfeng Zhao, Xia Mao, Lijiang Chen
- 来源：Biomedical Signal Processing and Control
- 链接：https://doi.org/10.1016/j.bspc.2018.08.035
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 812. Speech Recognition Using Deep Neural Networks: A Systematic Review
- 类型：语音/音频 / 系统/框架
- 标签：Speech, Framework/System
- 重要性：综述论文
- 引用数：996
- 年份：2019
- 作者：Ali Bou Nassif, Ismail Shahin, Imtinan Attili et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2019.2896880
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 813. fPINNs: Fractional Physics-Informed Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：994
- 年份：2019
- 作者：Guofei Pang, Lu Lu, George Em Karniadakis
- 来源：SIAM Journal on Scientific Computing
- 链接：https://doi.org/10.1137/18m1229845
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 814. Synaptic plasticity, memory and the hippocampus: a neural network approach to causality
- 类型：强化学习
- 标签：RL
- 重要性：综述论文
- 引用数：993
- 年份：2008
- 作者：Guilherme Neves, Sam F. Cooke, Tim V. P. Bliss
- 来源：Nature Reviews Neuroscience
- 链接：https://doi.org/10.1038/nrn2303
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 815. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：993
- 年份：2020
- 作者：Ameya D. Jagtap, Kenji Kawaguchi, George Em Karniadakis
- 来源：Journal of Computational Physics
- 链接：https://doi.org/10.1016/j.jcp.2019.109136
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 816. Going Deeper with Embedded FPGA Platform for Convolutional Neural Network
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：992
- 年份：2016
- 作者：Jiantao Qiu, Jie Wang, Song Yao et al.
- 来源：Proceedings of the 2016 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays
- 链接：https://doi.org/10.1145/2847263.2847265
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 817. CutPaste: Self-Supervised Learning for Anomaly Detection and Localization
- 类型：计算机视觉
- 标签：视觉, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：991
- 年份：2021
- 作者：Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00954
- 概述：研究自监督/对比学习，从无标注数据中学习可迁移表征。

## 818. Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：991
- 年份：2019
- 作者：Yu-Hsin Chen, Tien-Ju Yang, Joel S. Emer et al.
- 来源：IEEE Journal on Emerging and Selected Topics in Circuits and Systems
- 链接：https://doi.org/10.1109/jetcas.2019.2910232
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 819. Document Modeling with Gated Recurrent Neural Network for Sentiment Classification
- 类型：NLP/语言模型
- 标签：Language Model, RNN
- 重要性：架构论文
- 引用数：989
- 年份：2015
- 作者：Duyu Tang, Bing Qin, Ting Liu
- 来源：Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing
- 链接：https://doi.org/10.18653/v1/d15-1167
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 820. Reliable prediction of T‐cell epitopes using neural networks with novel sequence representations
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：989
- 年份：2003
- 作者：Morten Nielsen, Claus Lundegaard, Peder Worning et al.
- 来源：Protein Science
- 链接：https://doi.org/10.1110/ps.0239403
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 821. Neural network computation with DNA strand displacement cascades
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：987
- 年份：2011
- 作者：Lulu Qian, Erik Winfree, Jehoshua Bruck
- 来源：Nature
- 链接：https://doi.org/10.1038/nature10262
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 822. Bleeding complications of oral anticoagulant treatment: an inception-cohort, prospective collaborative study (ISCOAT)
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：987
- 年份：1996
- 作者：Gualtiero Palareti, Nicoletta Leali, Sergio Coccheri et al.
- 来源：The Lancet
- 链接：https://doi.org/10.1016/s0140-6736(96)01109-9
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 823. Going Deeper With Contextual CNN for Hyperspectral Image Classification
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：987
- 年份：2017
- 作者：Hyungtae Lee, Heesung Kwon
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2017.2725580
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 824. Knowledge Graph Convolutional Networks for Recommender Systems
- 类型：计算机视觉 / 图学习 / 系统/框架
- 标签：视觉, Graph, Framework/System, CNN
- 重要性：系统/框架
- 引用数：987
- 年份：2019
- 作者：Hongwei Wang, Miao Zhao, Xing Xie et al.
- 来源：The World Wide Web Conference
- 链接：https://doi.org/10.1145/3308558.3313417
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 825. Enhancing the Sensitivity of Frequency and Energy Splitting Detection by Using Exceptional Points: Application to Microcavity Sensors for Single-Particle Detection
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：综述论文
- 引用数：984
- 年份：2014
- 作者：Jan Wiersig
- 来源：Physical Review Letters
- 链接：https://doi.org/10.1103/physrevlett.112.203901
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 826. Multivariate LSTM-FCNs for time series classification
- 类型：深度学习相关
- 标签：RNN, Segmentation
- 重要性：奠基/方法论文
- 引用数：984
- 年份：2019
- 作者：Fazle Karim, Somshubra Majumdar, Houshang Darabi et al.
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2019.04.014
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 827. PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments, and Partial Charges
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：983
- 年份：2019
- 作者：Oliver T. Unke, Markus Meuwly
- 来源：Journal of Chemical Theory and Computation
- 链接：https://doi.org/10.1021/acs.jctc.9b00181
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 828. Nuclei Segmentation with Recurrent Residual Convolutional Neural Networks based U-Net (R2U-Net)
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection, CNN, RNN, Segmentation
- 重要性：奠基/方法论文
- 引用数：982
- 年份：2018
- 作者：Md Zahangir Alom, Chris Yakopcic, Tarek M. Taha et al.
- 来源：NAECON 2018 - IEEE National Aerospace and Electronics Conference
- 链接：https://doi.org/10.1109/naecon.2018.8556686
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 829. Deep Convolutional Network Cascade for Facial Point Detection
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：981
- 年份：2013
- 作者：Yi Sun, Xiaogang Wang, Xiaoou Tang
- 来源：2013 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2013.446
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 830. Apple detection during different growth stages in orchards using the improved YOLO-V3 model
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：应用/方法论文
- 引用数：981
- 年份：2019
- 作者：Yunong Tian, Guodong Yang, Zhe Wang et al.
- 来源：Computers and Electronics in Agriculture
- 链接：https://doi.org/10.1016/j.compag.2019.01.012
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 831. FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：981
- 年份：2019
- 作者：Bichen Wu, Kurt Keutzer, Xiaoliang Dai et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.01099
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 832. Quantitative analysis of protein far UV circular dichroism spectra by neural networks
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：980
- 年份：1992
- 作者：Gerald Böhm, Rudolf Muhr, Rainer Jaenicke
- 来源："Protein Engineering, Design and Selection"
- 链接：https://doi.org/10.1093/protein/5.3.191
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 833. Improved Automated Detection of Diabetic Retinopathy on a Publicly Available Dataset Through Integration of Deep Learning
- 类型：计算机视觉
- 标签：视觉, Dataset/Benchmark
- 重要性：数据集/基准
- 引用数：979
- 年份：2016
- 作者：Michael David Abràmoff, Yiyue Lou, Ali Erginay et al.
- 来源：Investigative Opthalmology &amp;amp; Visual Science
- 链接：https://doi.org/10.1167/iovs.16-19964
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 834. Efficient object localization using Convolutional Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：979
- 年份：2015
- 作者：Jonathan Tompson, Ross Goroshin, Arjun Jain et al.
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7298664
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 835. Deep learning on image denoising: An overview
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：综述论文
- 引用数：979
- 年份：2020
- 作者：Chunwei Tian, Lunke Fei, Wenxian Zheng et al.
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2020.07.025
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 836. LSTM neural networks for language modeling
- 类型：NLP/语言模型 / 语音/音频
- 标签：Language Model, Speech, RNN
- 重要性：奠基/方法论文
- 引用数：979
- 年份：2012
- 作者：Martin Sundermeyer, Ralf Schlüter, Hermann Ney
- 来源：Interspeech 2012
- 链接：https://doi.org/10.21437/interspeech.2012-65
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 837. 30-W/mm GaN HEMTs by Field Plate Optimization
- 类型：生成模型
- 标签：Generative AI, Optimization/Training, GAN
- 重要性：应用/方法论文
- 引用数：977
- 年份：2004
- 作者：Y.-F. Wu, A. Saxler, M. Moore et al.
- 来源：IEEE Electron Device Letters
- 链接：https://doi.org/10.1109/led.2003.822667
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 838. Deep Learning for Brain MRI Segmentation: State of the Art and Future Directions
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：应用/方法论文
- 引用数：976
- 年份：2017
- 作者：Zeynettin Akkus, Alfiia Galimzianova, Assaf Hoogi et al.
- 来源：Journal of Digital Imaging
- 链接：https://doi.org/10.1007/s10278-017-9983-4
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 839. Flamingo: A Visual Language Model for Few-Shot Learning
- 类型：NLP/语言模型 / 计算机视觉 / 系统/框架
- 标签：Language Model, 视觉, Framework/System
- 重要性：系统/框架
- 引用数：975
- 年份：2022
- 作者：Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc et al.
- 来源：Advances in Neural Information Processing Systems 35
- 链接：https://doi.org/10.52202/068431-1723
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 840. Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：973
- 年份：2020
- 作者：S. Mostafa Mousavi, William L. Ellsworth, Weiqiang Zhu et al.
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/s41467-020-17591-w
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 841. Artificial Neural Networks-Based Machine Learning for Wireless Networks: A Tutorial
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：971
- 年份：2019
- 作者：Mingzhe Chen, Ursula Challita, Walid Saad et al.
- 来源：IEEE Communications Surveys &amp;amp; Tutorials
- 链接：https://doi.org/10.1109/comst.2019.2926625
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 842. Automated Melanoma Recognition in Dermoscopy Images via Very Deep Residual Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：970
- 年份：2017
- 作者：Lequan Yu, Hao Chen, Qi Dou et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2642839
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 843. Convolutional Neural Networks for Large-Scale Remote-Sensing Image Classification
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：970
- 年份：2017
- 作者：Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2016.2612821
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 844. Mask Scoring R-CNN
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：969
- 年份：2019
- 作者：Zhaojin Huang, Lichao Huang, Yongchao Gong et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00657
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 845. Monocular 3D Human Pose Estimation in the Wild Using Improved CNN Supervision
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：969
- 年份：2017
- 作者：Dushyant Mehta, Helge Rhodin, Dan Casas et al.
- 来源：2017 International Conference on 3D Vision (3DV)
- 链接：https://doi.org/10.1109/3dv.2017.00064
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 846. Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：968
- 年份：2018
- 作者：Weilin Xu, David Evans, Yanjun Qi
- 来源：Proceedings 2018 Network and Distributed System Security Symposium
- 链接：https://doi.org/10.14722/ndss.2018.23198
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 847. A convolutional neural network cascade for face detection
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：967
- 年份：2015
- 作者：Haoxiang Li, Zhe Lin, Xiaohui Shen et al.
- 来源：2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2015.7299170
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 848. DeepFruits: A Fruit Detection System Using Deep Neural Networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：967
- 年份：2016
- 作者：Inkyu Sa, Zongyuan Ge, Feras Dayoub et al.
- 来源：Sensors
- 链接：https://doi.org/10.3390/s16081222
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 849. The power of quantum neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：966
- 年份：2021
- 作者：Amira Abbas, David Sutter, Christa Zoufal et al.
- 来源：Nature Computational Science
- 链接：https://doi.org/10.1038/s43588-021-00084-1
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 850. Deep Residual Learning for Image Recognition: A Survey
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：综述论文
- 引用数：962
- 年份：2022
- 作者：Muhammad Shafiq, Zhaoquan Gu
- 来源：Applied Sciences
- 链接：https://doi.org/10.3390/app12188972
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 851. Gate Injection Transistor (GIT)—A Normally-Off AlGaN/GaN Power Transistor Using Conductivity Modulation
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：961
- 年份：2007
- 作者：Yasuhiro Uemoto, Masahiro Hikita, Hiroaki Ueno et al.
- 来源：IEEE Transactions on Electron Devices
- 链接：https://doi.org/10.1109/ted.2007.908601
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 852. Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Dataset/Benchmark, Segmentation
- 重要性：综述论文
- 引用数：959
- 年份：2020
- 作者：Nima Tajbakhsh, Laura Jeyaseelan, Qian Li et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2020.101693
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 853. Dense Nested Attention Network for Infrared Small Target Detection
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：958
- 年份：2023
- 作者：Boyang Li, Chao Xiao, Longguang Wang et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2022.3199107
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 854. SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：架构论文
- 引用数：957
- 年份：2022
- 作者：Sheng Fang, Kaiyu Li, Jinyuan Shao et al.
- 来源：IEEE Geoscience and Remote Sensing Letters
- 链接：https://doi.org/10.1109/lgrs.2021.3056416
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 855. Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：956
- 年份：2017
- 作者：Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon et al.
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.240
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 856. Artificial neural networks in renewable energy systems applications: a review
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：955
- 年份：2001
- 作者：Soteris A. Kalogirou
- 来源：Renewable and Sustainable Energy Reviews
- 链接：https://doi.org/10.1016/s1364-0321(01)00006-5
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 857. Machine Learning and Deep Learning Methods for Intrusion Detection Systems: A Survey
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：954
- 年份：2019
- 作者：Hongyu Liu, Bo Lang
- 来源：Applied Sciences
- 链接：https://doi.org/10.3390/app9204396
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 858. Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：954
- 年份：2019
- 作者：Yang He, Ping Liu, Ziwei Wang et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00447
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 859. Elastic properties of zinc-blende and wurtzite AlN, GaN, and InN
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：953
- 年份：1997
- 作者：A. F. Wright
- 来源：Journal of Applied Physics
- 链接：https://doi.org/10.1063/1.366114
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 860. Spin-glass models of neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：953
- 年份：1985
- 作者：Daniel J. Amit, Hanoch Gutfreund, H. Sompolinsky
- 来源：Physical Review A
- 链接：https://doi.org/10.1103/physreva.32.1007
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 861. Automated Breast Ultrasound Lesions Detection Using Convolutional Neural Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：951
- 年份：2018
- 作者：Moi Hoon Yap, Gerard Pons, Joan Martí et al.
- 来源：IEEE Journal of Biomedical and Health Informatics
- 链接：https://doi.org/10.1109/jbhi.2017.2731873
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 862. Reading Text in the Wild with Convolutional Neural Networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：950
- 年份：2016
- 作者：Max Jaderberg, Karen Simonyan, Andrea Vedaldi et al.
- 来源：International Journal of Computer Vision
- 链接：https://doi.org/10.1007/s11263-015-0823-z
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 863. Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：948
- 年份：1985
- 作者：Daniel J. Amit, Hanoch Gutfreund, H. Sompolinsky
- 来源：Physical Review Letters
- 链接：https://doi.org/10.1103/physrevlett.55.1530
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 864. Experimental demonstration of associative memory with memristive neural networks
- 类型：深度学习相关
- 标签：Segmentation
- 重要性：架构论文
- 引用数：945
- 年份：2010
- 作者：Yuriy V. Pershin, Massimiliano Di Ventra
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/j.neunet.2010.05.001
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 865. Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：945
- 年份：2020
- 作者：Ameya D. Jagtap, George Em Karniadakis
- 来源：Communications in Computational Physics
- 链接：https://doi.org/10.4208/cicp.oa-2020-0164
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 866. Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
- 类型：计算机视觉
- 标签：视觉, RNN
- 重要性：架构论文
- 引用数：943
- 年份：2016
- 作者：Sean Bell, C. Lawrence Zitnick, Kavita Bala et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.314
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 867. The imperative for regulatory oversight of large language models (or generative AI) in healthcare
- 类型：NLP/语言模型 / 生成模型 / 医疗/生命科学AI
- 标签：Language Model, Generative AI, Medical AI, Attention/Transformer
- 重要性：应用/方法论文
- 引用数：942
- 年份：2023
- 作者：Bertalan Meskó, Eric J. Topol
- 来源：npj Digital Medicine
- 链接：https://doi.org/10.1038/s41746-023-00873-0
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 868. Super-convergence: very fast training of neural networks using large learning rates
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：941
- 年份：2019
- 作者：Leslie N. Smith, Nicholay Topin
- 来源：Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications
- 链接：https://doi.org/10.1117/12.2520589
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 869. CNN-RNN: A Unified Framework for Multi-label Image Classification
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, CNN, RNN
- 重要性：系统/框架
- 引用数：940
- 年份：2016
- 作者：Jiang Wang, Yi Yang, Junhua Mao et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.251
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 870. Directed differentiation of human pluripotent stem cells to cerebral cortex neurons and neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：939
- 年份：2012
- 作者：Yichen Shi, Peter Kirwan, Frederick J Livesey
- 来源：Nature Protocols
- 链接：https://doi.org/10.1038/nprot.2012.116
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 871. Hybrid computing using a neural network with dynamic external memory
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：938
- 年份：2016
- 作者：Alex Graves, Greg Wayne, Malcolm Reynolds et al.
- 来源：Nature
- 链接：https://doi.org/10.1038/nature20101
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 872. A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：综述论文
- 引用数：937
- 年份：2023
- 作者：Jie Gui, Zhenan Sun, Yonggang Wen et al.
- 来源：IEEE Transactions on Knowledge and Data Engineering
- 链接：https://doi.org/10.1109/tkde.2021.3130191
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 873. The ART of adaptive pattern recognition by a self-organizing neural network
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：936
- 年份：1988
- 作者：G.A. Carpenter, S. Grossberg
- 来源：Computer
- 链接：https://doi.org/10.1109/2.33
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 874. Plant Disease Detection and Classification by Deep Learning—A Review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：936
- 年份：2021
- 作者：Lili Li, Shujuan Zhang, Bin Wang
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2021.3069646
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 875. Multilabel Neural Networks with Applications to Functional Genomics and Text Categorization
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：935
- 年份：2006
- 作者：Min-Ling Zhang, Zhi-Hua Zhou
- 来源：IEEE Transactions on Knowledge and Data Engineering
- 链接：https://doi.org/10.1109/tkde.2006.162
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 876. Learning From Noisy Labels With Deep Neural Networks: A Survey
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：933
- 年份：2023
- 作者：Hwanjun Song, Minseok Kim, Dongmin Park et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2022.3152527
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 877. Deep Residual Network for Steganalysis of Digital Images
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：931
- 年份：2019
- 作者：Mehdi Boroumand, Mo Chen, Jessica Fridrich
- 来源：IEEE Transactions on Information Forensics and Security
- 链接：https://doi.org/10.1109/tifs.2018.2871749
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 878. Chaos in Random Neural Networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：931
- 年份：1988
- 作者：H. Sompolinsky, A. Crisanti, H. J. Sommers
- 来源：Physical Review Letters
- 链接：https://doi.org/10.1103/physrevlett.61.259
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 879. Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System
- 重要性：系统/框架
- 引用数：929
- 年份：2017
- 作者：Jeffrey Mahler, Jacky Liang, Sherdil Niyaz et al.
- 来源：Robotics: Science and Systems XIII
- 链接：https://doi.org/10.15607/rss.2017.xiii.058
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 880. Predicting the secondary structure of globular proteins using neural network models
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：928
- 年份：1988
- 作者：Ning Qian, Terrence J. Sejnowski
- 来源：Journal of Molecular Biology
- 链接：https://doi.org/10.1016/0022-2836(88)90564-5
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 881. Generative Adversarial Networks for Noise Reduction in Low-Dose CT
- 类型：生成模型 / 医疗/生命科学AI
- 标签：Generative AI, Medical AI, GAN
- 重要性：架构论文
- 引用数：925
- 年份：2017
- 作者：Jelmer M. Wolterink, Tim Leiner, Max A. Viergever et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2017.2708987
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 882. SPIKING NEURAL NETWORKS
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：924
- 年份：2009
- 作者：SAMANWOY GHOSH-DASTIDAR, HOJJAT ADELI
- 来源：International Journal of Neural Systems
- 链接：https://doi.org/10.1142/s0129065709002002
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 883. A Review of GaN on SiC High Electron-Mobility Power Transistors and MMICs
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：综述论文
- 引用数：923
- 年份：2012
- 作者：Raymond S. Pengelly, Simon M. Wood, James W. Milligan et al.
- 来源：IEEE Transactions on Microwave Theory and Techniques
- 链接：https://doi.org/10.1109/tmtt.2012.2187535
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 884. Deep Neural Networks Reveal a Gradient in the Complexity of Neural Representations across the Ventral Stream
- 类型：深度学习相关
- 标签：Optimization/Training
- 重要性：架构论文
- 引用数：922
- 年份：2015
- 作者：U. Guclu, M. A. J. van Gerven
- 来源：Journal of Neuroscience
- 链接：https://doi.org/10.1523/jneurosci.5023-14.2015
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 885. Image denoising: Can plain neural networks compete with BM3D?
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：922
- 年份：2012
- 作者：H. C. Burger, C. J. Schuler, S. Harmeling
- 来源：2012 IEEE Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2012.6247952
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 886. Recent advances in deep learning for object detection
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：应用/方法论文
- 引用数：920
- 年份：2020
- 作者：Xiongwei Wu, Doyen Sahoo, Steven C.H. Hoi
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2020.01.085
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 887. Phase recovery and holographic image reconstruction using deep learning in neural networks
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：919
- 年份：2017
- 作者：Yair Rivenson, Yibo Zhang, Harun Günaydın et al.
- 来源：Light: Science &amp;amp; Applications
- 链接：https://doi.org/10.1038/lsa.2017.141
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 888. Identification of rice diseases using deep convolutional neural networks
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：917
- 年份：2017
- 作者：Yang Lu, Shujuan Yi, Nianyin Zeng et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2017.06.023
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 889. Deep convolutional neural network based medical image classification for disease diagnosis
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：916
- 年份：2019
- 作者：Samir S. Yadav, Shivajirao M. Jadhav
- 来源：Journal of Big Data
- 链接：https://doi.org/10.1186/s40537-019-0276-2
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 890. Hole Compensation Mechanism of P-Type GaN Films
- 类型：生成模型
- 标签：Generative AI, GAN
- 重要性：应用/方法论文
- 引用数：915
- 年份：1992
- 作者：Shuji Nakamura, Naruhito Iwasa, Masayuki Senoh Masayuki Senoh et al.
- 来源：Japanese Journal of Applied Physics
- 链接：https://doi.org/10.1143/jjap.31.1258
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 891. Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：915
- 年份：2017
- 作者：Martin Simonovsky, Nikos Komodakis
- 来源：2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2017.11
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 892. Deep Neural Networks: A New Framework for Modeling Biological Vision and Brain Information Processing
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System
- 重要性：综述论文
- 引用数：915
- 年份：2015
- 作者：Nikolaus Kriegeskorte
- 来源：Annual Review of Vision Science
- 链接：https://doi.org/10.1146/annurev-vision-082114-035447
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 893. Extensions of recurrent neural network language model
- 类型：NLP/语言模型 / 语音/音频
- 标签：Language Model, Speech, RNN
- 重要性：架构论文
- 引用数：913
- 年份：2011
- 作者：Tomas Mikolov, Stefan Kombrink, Lukas Burget et al.
- 来源：2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- 链接：https://doi.org/10.1109/icassp.2011.5947611
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 894. Deep learning algorithms for detection of critical findings in head CT scans: a retrospective study
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：911
- 年份：2018
- 作者：Sasank Chilamkurthy, Rohit Ghosh, Swetha Tanamala et al.
- 来源：The Lancet
- 链接：https://doi.org/10.1016/s0140-6736(18)31645-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 895. DAGAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction
- 类型：生成模型 / 医疗/生命科学AI
- 标签：Generative AI, Medical AI, GAN
- 重要性：架构论文
- 引用数：910
- 年份：2018
- 作者：Guang Yang, Simiao Yu, Hao Dong et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2017.2785879
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 896. A review of wind speed and wind power forecasting with deep neural networks
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：910
- 年份：2021
- 作者：Yun Wang, Runmin Zou, Fang Liu et al.
- 来源：Applied Energy
- 链接：https://doi.org/10.1016/j.apenergy.2021.117766
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 897. TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Object Detection
- 重要性：架构论文
- 引用数：906
- 年份：2022
- 作者：Xuyang Bai, Zeyu Hu, Xinge Zhu et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00116
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 898. Visual attention network
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：905
- 年份：2023
- 作者：Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu et al.
- 来源：Computational Visual Media
- 链接：https://doi.org/10.1007/s41095-023-0364-2
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 899. UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Residual/Skip Connection, Segmentation
- 重要性：架构论文
- 引用数：904
- 年份：2022
- 作者：Haonan Wang, Peng Cao, Jiaqi Wang et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v36i3.20144
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 900. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：应用/方法论文
- 引用数：902
- 年份：2022
- 作者：Noah F. Greenwald, Geneva Miller, Erick Moen et al.
- 来源：Nature Biotechnology
- 链接：https://doi.org/10.1038/s41587-021-01094-0
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 901. Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection
- 重要性：架构论文
- 引用数：900
- 年份：2021
- 作者：Jiajun Deng, Shaoshuai Shi, Peiwei Li et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v35i2.16207
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 902. SA-Net: Shuffle Attention for Deep Convolutional Neural Networks
- 类型：计算机视觉 / 语音/音频
- 标签：视觉, Speech, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：899
- 年份：2021
- 作者：Qing-Long Zhang, Yu-Bin Yang
- 来源：ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
- 链接：https://doi.org/10.1109/icassp39728.2021.9414568
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 903. An accurate comparison of methods for quantifying variable importance in artificial neural networks using simulated data
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：模型/技术报告
- 引用数：898
- 年份：2004
- 作者：Julian D Olden, Michael K Joy, Russell G Death
- 来源：Ecological Modelling
- 链接：https://doi.org/10.1016/j.ecolmodel.2004.03.013
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 904. &lt;i&gt;K&lt;/i&gt; &lt;sub&gt;DEEP&lt;/sub&gt; : Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：895
- 年份：2018
- 作者：José Jiménez, Miha Škalič, Gerard Martínez-Rosell et al.
- 来源：Journal of Chemical Information and Modeling
- 链接：https://doi.org/10.1021/acs.jcim.7b00650
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 905. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：890
- 年份：2023
- 作者：Bowen Deng, Peichen Zhong, KyuJung Jun et al.
- 来源：Nature Machine Intelligence
- 链接：https://doi.org/10.1038/s42256-023-00716-3
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 906. Deepfake Video Detection Using Recurrent Neural Networks
- 类型：视频/世界模型
- 标签：Video, RNN
- 重要性：架构论文
- 引用数：889
- 年份：2018
- 作者：David Guera, Edward J. Delp
- 来源：2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)
- 链接：https://doi.org/10.1109/avss.2018.8639163
- 概述：研究循环神经网络及门控单元，用于序列建模和长期依赖学习。

## 907. Probable networks and plausible predictions — a review of practical Bayesian methods for supervised neural networks
- 类型：系统/框架
- 标签：Framework/System
- 重要性：综述论文
- 引用数：888
- 年份：1995
- 作者：David J C Mackay
- 来源：Network: Computation in Neural Systems
- 链接：https://doi.org/10.1088/0954-898x_6_3_011
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 908. The space of interactions in neural network models
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：881
- 年份：1988
- 作者：E Gardner
- 来源：Journal of Physics A: Mathematical and General
- 链接：https://doi.org/10.1088/0305-4470/21/1/030
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 909. Blind Image Quality Assessment Using a Deep Bilinear Convolutional Neural Network
- 类型：计算机视觉 / 视频/世界模型 / 系统/框架
- 标签：视觉, Video, Framework/System, CNN
- 重要性：系统/框架
- 引用数：881
- 年份：2020
- 作者：Weixia Zhang, Kede Ma, Jia Yan et al.
- 来源：IEEE Transactions on Circuits and Systems for Video Technology
- 链接：https://doi.org/10.1109/tcsvt.2018.2886771
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 910. Evaluation of secondary structure of proteins from UV circular dichroism spectra using an unsupervised learning neural network
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：架构论文
- 引用数：881
- 年份：1993
- 作者：M.A. Andrade, P. Chacón, J.J. Merelo et al.
- 来源："Protein Engineering, Design and Selection"
- 链接：https://doi.org/10.1093/protein/6.4.383
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 911. Detection and diagnosis of dental caries using a deep learning-based convolutional neural network algorithm
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：881
- 年份：2018
- 作者：Jae-Hong Lee, Do-Hyung Kim, Seong-Nyum Jeong et al.
- 来源：Journal of Dentistry
- 链接：https://doi.org/10.1016/j.jdent.2018.07.015
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 912. Person Re-identification by Multi-Channel Parts-Based CNN with Improved Triplet Loss Function
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：881
- 年份：2016
- 作者：De Cheng, Yihong Gong, Sanping Zhou et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.149
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 913. Neural network of cognitive emotion regulation — An ALE meta-analysis and MACM analysis
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：880
- 年份：2014
- 作者：N. Kohn, S.B. Eickhoff, M. Scheller et al.
- 来源：NeuroImage
- 链接：https://doi.org/10.1016/j.neuroimage.2013.11.001
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 914. NB-CNN: Deep Learning-Based Crack Detection Using Convolutional Neural Network and Naïve Bayes Data Fusion
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：879
- 年份：2018
- 作者：Fu-Chen Chen, Mohammad R. Jahanshahi
- 来源：IEEE Transactions on Industrial Electronics
- 链接：https://doi.org/10.1109/tie.2017.2764844
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 915. Spatial as Deep: Spatial CNN for Traffic Scene Understanding
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：879
- 年份：2018
- 作者：Xingang Pan, Jianping Shi, Ping Luo et al.
- 来源：Proceedings of the AAAI Conference on Artificial Intelligence
- 链接：https://doi.org/10.1609/aaai.v32i1.12301
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 916. Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, CNN
- 重要性：架构论文
- 引用数：878
- 年份：2020
- 作者：Abduallah Mohamed, Kun Qian, Mohamed Elhoseiny et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.01443
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 917. Medical Image Segmentation Review: The Success of U-Net
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：综述论文
- 引用数：878
- 年份：2024
- 作者：Reza Azad, Ehsan Khodapanah Aghdam, Amelie Rauland et al.
- 来源：IEEE Transactions on Pattern Analysis and Machine Intelligence
- 链接：https://doi.org/10.1109/tpami.2024.3435571
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 918. Plant diseases and pests detection based on deep learning: a review
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：877
- 年份：2021
- 作者：Jun Liu, Xuewei Wang
- 来源：Plant Methods
- 链接：https://doi.org/10.1186/s13007-021-00722-9
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 919. Evaluating the Visualization of What a Deep Neural Network Has Learned
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System
- 重要性：系统/框架
- 引用数：876
- 年份：2017
- 作者：Wojciech Samek, Alexander Binder, Gregoire Montavon et al.
- 来源：IEEE Transactions on Neural Networks and Learning Systems
- 链接：https://doi.org/10.1109/tnnls.2016.2599820
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 920. TrackFormer: Multi-Object Tracking with Transformers
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：876
- 年份：2022
- 作者：Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00864
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 921. VulDeePecker: A Deep Learning-Based System for Vulnerability Detection
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：875
- 年份：2018
- 作者：Zhen Li, Deqing Zou, Shouhuai Xu et al.
- 来源：Proceedings 2018 Network and Distributed System Security Symposium
- 链接：https://doi.org/10.14722/ndss.2018.23158
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 922. Distributed attack detection scheme using deep learning approach for Internet of Things
- 类型：系统/框架
- 标签：Framework/System
- 重要性：系统/框架
- 引用数：873
- 年份：2018
- 作者：Abebe Abeshu Diro, Naveen Chilamkurti
- 来源：Future Generation Computer Systems
- 链接：https://doi.org/10.1016/j.future.2017.08.043
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 923. Skeleton-Based Action Recognition With Shift Graph Convolutional Network
- 类型：计算机视觉 / 视频/世界模型 / 图学习
- 标签：视觉, Video, Graph, CNN
- 重要性：架构论文
- 引用数：870
- 年份：2020
- 作者：Ke Cheng, Yifan Zhang, Xiangyu He et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.00026
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 924. Pruning and quantization for deep neural network acceleration: A survey
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：综述论文
- 引用数：870
- 年份：2021
- 作者：Tailin Liang, John Glossner, Lei Wang et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2021.07.045
- 概述：综述深度学习相关方向的方法、应用、挑战与发展趋势。

## 925. Deep Convolutional Neural Networks with transfer learning for computer vision-based data-driven pavement distress detection
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：867
- 年份：2017
- 作者：Kasthurirangan Gopalakrishnan, Siddhartha K. Khaitan, Alok Choudhary et al.
- 来源：Construction and Building Materials
- 链接：https://doi.org/10.1016/j.conbuildmat.2017.09.110
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 926. A Light CNN for Deep Face Representation With Noisy Labels
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：860
- 年份：2018
- 作者：Xiang Wu, Ran He, Zhenan Sun et al.
- 来源：IEEE Transactions on Information Forensics and Security
- 链接：https://doi.org/10.1109/tifs.2018.2833032
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 927. Dynamics of the Immune Reaction to Pancreatic Cancer from Inception to Invasion
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN
- 重要性：架构论文
- 引用数：858
- 年份：2007
- 作者：Carolyn E. Clark, Sunil R. Hingorani, Rosemarie Mick et al.
- 来源：Cancer Research
- 链接：https://doi.org/10.1158/0008-5472.can-07-0175
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 928. Multi-attentional Deepfake Detection
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：855
- 年份：2021
- 作者：Hanqing Zhao, Tianyi Wei, Wenbo Zhou et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00222
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 929. Neocognitron: A hierarchical neural network capable of visual pattern recognition
- 类型：计算机视觉
- 标签：视觉
- 重要性：架构论文
- 引用数：852
- 年份：1988
- 作者：Kunihiko Fukushima
- 来源：Neural Networks
- 链接：https://doi.org/10.1016/0893-6080(88)90014-7
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 930. A survey of modern deep learning based object detection models
- 类型：计算机视觉
- 标签：视觉, Object Detection
- 重要性：综述论文
- 引用数：851
- 年份：2022
- 作者：Syed Sahil Abbas Zaidi, Mohammad Samar Ansari, Asra Aslam et al.
- 来源：Digital Signal Processing
- 链接：https://doi.org/10.1016/j.dsp.2022.103514
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 931. Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：849
- 年份：2020
- 作者：En Li, Liekang Zeng, Zhi Zhou et al.
- 来源：IEEE Transactions on Wireless Communications
- 链接：https://doi.org/10.1109/twc.2019.2946140
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 932. SEGAN: Speech Enhancement Generative Adversarial Network
- 类型：生成模型 / 语音/音频
- 标签：Generative AI, Speech, GAN
- 重要性：架构论文
- 引用数：849
- 年份：2017
- 作者：Santiago Pascual, Antonio Bonafonte, Joan Serrà
- 来源：Interspeech 2017
- 链接：https://doi.org/10.21437/interspeech.2017-1428
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 933. Deep Learning to Improve Breast Cancer Detection on Screening Mammography
- 类型：医疗/生命科学AI
- 标签：Medical AI
- 重要性：应用/方法论文
- 引用数：849
- 年份：2019
- 作者：Li Shen, Laurie R. Margolies, Joseph H. Rothstein et al.
- 来源：Scientific Reports
- 链接：https://doi.org/10.1038/s41598-019-48995-4
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 934. MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding
- 类型：图学习
- 标签：Graph
- 重要性：架构论文
- 引用数：848
- 年份：2020
- 作者：Xinyu Fu, Jiani Zhang, Ziqiao Meng et al.
- 来源：Proceedings of The Web Conference 2020
- 链接：https://doi.org/10.1145/3366423.3380297
- 概述：研究图神经网络或图表示学习，用神经模型处理节点、边和图结构数据。

## 935. Automatic diagnosis of the 12-lead ECG using a deep neural network
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：架构论文
- 引用数：848
- 年份：2020
- 作者：Antônio H. Ribeiro, Manoel Horta Ribeiro, Gabriela M. M. Paixão et al.
- 来源：Nature Communications
- 链接：https://doi.org/10.1038/s41467-020-15432-4
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 936. Large scale deep learning for computer aided detection of mammographic lesions
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI
- 重要性：应用/方法论文
- 引用数：847
- 年份：2017
- 作者：Thijs Kooi, Geert Litjens, Bram van Ginneken et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2016.07.007
- 概述：深度学习相关高被引研究，常作为相关方法、模型、数据集或应用工作的参考。

## 937. Learning Texture Transformer Network for Image Super-Resolution
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer
- 重要性：架构论文
- 引用数：845
- 年份：2020
- 作者：Fuzhi Yang, Huan Yang, Jianlong Fu et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.00583
- 概述：研究注意力或Transformer结构，提升序列、视觉或多模态建模能力。

## 938. Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud
- 类型：计算机视觉 / 图学习
- 标签：视觉, Graph, Object Detection
- 重要性：架构论文
- 引用数：844
- 年份：2020
- 作者：Weijing Shi, Raj Rajkumar
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.00178
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 939. Plant leaf disease classification using EfficientNet deep learning model
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：应用/方法论文
- 引用数：828
- 年份：2021
- 作者：Ümit Atila, Murat Uçar, Kemal Akyol et al.
- 来源：Ecological Informatics
- 链接：https://doi.org/10.1016/j.ecoinf.2020.101182
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 940. BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：814
- 年份：2015
- 作者：Jifeng Dai, Kaiming He, Jian Sun
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.191
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 941. FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：814
- 年份：2019
- 作者：Xinming Wu, Luming Liang, Yunzhi Shi et al.
- 来源：Geophysics
- 链接：https://doi.org/10.1190/geo2018-0646.1
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 942. Deep Learning for Cardiac Image Segmentation: A Review
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：综述论文
- 引用数：814
- 年份：2020
- 作者：Chen Chen, Chen Qin, Huaqi Qiu et al.
- 来源：Frontiers in Cardiovascular Medicine
- 链接：https://doi.org/10.3389/fcvm.2020.00025
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 943. A review of semantic segmentation using deep neural networks
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：综述论文
- 引用数：790
- 年份：2018
- 作者：Yanming Guo, Yu Liu, Theodoros Georgiou et al.
- 来源：International Journal of Multimedia Information Retrieval
- 链接：https://doi.org/10.1007/s13735-017-0141-z
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 944. Computer vision-based concrete crack detection using U-net fully convolutional networks
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：790
- 年份：2019
- 作者：Zhenqing Liu, Yiwen Cao, Yize Wang et al.
- 来源：Automation in Construction
- 链接：https://doi.org/10.1016/j.autcon.2019.04.005
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 945. Medical image segmentation using deep learning: A survey
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：综述论文
- 引用数：787
- 年份：2022
- 作者：Risheng Wang, Tao Lei, Ruixia Cui et al.
- 来源：IET Image Processing
- 链接：https://doi.org/10.1049/ipr2.12419
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 946. ScribbleSup: Scribble-Supervised Convolutional Networks for Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：776
- 年份：2016
- 作者：Di Lin, Jifeng Dai, Jiaya Jia et al.
- 来源：2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2016.344
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 947. Recurrent residual U-Net for medical image segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Residual/Skip Connection, RNN, Segmentation
- 重要性：奠基/方法论文
- 引用数：760
- 年份：2019
- 作者：Md Zahangir Alom, Chris Yakopcic, Mahmudul Hasan et al.
- 来源：Journal of Medical Imaging
- 链接：https://doi.org/10.1117/1.jmi.6.1.014006
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 948. SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud
- 类型：计算机视觉
- 标签：视觉, CNN, RNN, Segmentation
- 重要性：应用/方法论文
- 引用数：758
- 年份：2018
- 作者：Bichen Wu, Alvin Wan, Xiangyu Yue et al.
- 来源：2018 IEEE International Conference on Robotics and Automation (ICRA)
- 链接：https://doi.org/10.1109/icra.2018.8462926
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 949. Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：架构论文
- 引用数：744
- 年份：2019
- 作者：Chenxi Liu, Liang-Chieh Chen, Florian Schroff et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00017
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 950. Automatic Segmentation of MR Brain Images With a Convolutional Neural Network
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：735
- 年份：2016
- 作者：Pim Moeskops, Max A. Viergever, Adrienne M. Mendrik et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2016.2548501
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 951. Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Dataset/Benchmark, Segmentation
- 重要性：数据集/基准
- 引用数：735
- 年份：2019
- 作者：Sarah Leclerc, Erik Smistad, Joao Pedrosa et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2019.2900516
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 952. A review of deep learning methods for semantic segmentation of remote sensing imagery
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, Segmentation
- 重要性：综述论文
- 引用数：731
- 年份：2021
- 作者：Xiaohui Yuan, Jianfang Shi, Lichuan Gu
- 来源：Expert Systems with Applications
- 链接：https://doi.org/10.1016/j.eswa.2020.114417
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 953. A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：730
- 年份：2019
- 作者：Nabila Abraham, Naimul Mefraz Khan
- 来源：2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019)
- 链接：https://doi.org/10.1109/isbi.2019.8759329
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 954. Weakly-and Semi-Supervised Learning of a Deep Convolutional Network for Semantic Image Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：724
- 年份：2015
- 作者：George Papandreou, Liang-Chieh Chen, Kevin P. Murphy et al.
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.203
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 955. Interactive Medical Image Segmentation Using Deep Learning With Image-Specific Fine Tuning
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：应用/方法论文
- 引用数：720
- 年份：2018
- 作者：Guotai Wang, Wenqi Li, Maria A. Zuluaga et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2018.2791721
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 956. CA-Net: Comprehensive Attention Convolutional Neural Networks for Explainable Medical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, CNN, Segmentation
- 重要性：架构论文
- 引用数：715
- 年份：2021
- 作者：Ran Gu, Guotai Wang, Tao Song et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2020.3035253
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 957. Deep convolutional neural networks for multi-modality isointense infant brain image segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：714
- 年份：2015
- 作者：Wenlu Zhang, Rongjian Li, Houtao Deng et al.
- 来源：NeuroImage
- 链接：https://doi.org/10.1016/j.neuroimage.2014.12.061
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 958. A deep learning model integrating FCNNs and CRFs for brain tumor segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：架构论文
- 引用数：712
- 年份：2018
- 作者：Xiaomei Zhao, Yihong Wu, Guidong Song et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2017.10.002
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 959. Classification of Skin Disease Using Deep Learning Neural Networks with MobileNet V2 and LSTM
- 类型：计算机视觉
- 标签：视觉, CNN, RNN
- 重要性：奠基/方法论文
- 引用数：697
- 年份：2021
- 作者：Parvathaneni Naga Srinivasu, Jalluri Gnana SivaSai, Muhammad Fazal Ijaz et al.
- 来源：Sensors
- 链接：https://doi.org/10.3390/s21082852
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 960. DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI / 系统/框架
- 标签：视觉, Medical AI, Framework/System, CNN, Segmentation
- 重要性：系统/框架
- 引用数：683
- 年份：2020
- 作者：Debesh Jha, Michael A. Riegler, Dag Johansen et al.
- 来源：2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS)
- 链接：https://doi.org/10.1109/cbms49503.2020.00111
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 961. Self-Supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Self-supervised Learning, Segmentation
- 重要性：应用/方法论文
- 引用数：668
- 年份：2020
- 作者：Yude Wang, Jie Zhang, Meina Kan et al.
- 来源：2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr42600.2020.01229
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 962. nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：666
- 年份：2023
- 作者：Hong-Yu Zhou, Jiansen Guo, Yinghao Zhang et al.
- 来源：IEEE Transactions on Image Processing
- 链接：https://doi.org/10.1109/tip.2023.3293771
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 963. Google Earth Engine Applications Since Inception: Usage, Trends, and Potential
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：652
- 年份：2018
- 作者：Lalit Kumar, Onisimo Mutanga
- 来源：Remote Sensing
- 链接：https://doi.org/10.3390/rs10101509
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 964. Resonance-stabilized hydrocarbon-radical chain reactions may explain soot inception and growth
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：647
- 年份：2018
- 作者：K. O. Johansson, M. P. Head-Gordon, P. E. Schrader et al.
- 来源：Science
- 链接：https://doi.org/10.1126/science.aat3417
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 965. VoxResNet: Deep voxelwise residual networks for brain segmentation from 3D MR images
- 类型：计算机视觉
- 标签：视觉, Residual/Skip Connection, Segmentation
- 重要性：奠基/方法论文
- 引用数：634
- 年份：2018
- 作者：Hao Chen, Qi Dou, Lequan Yu et al.
- 来源：NeuroImage
- 链接：https://doi.org/10.1016/j.neuroimage.2017.04.041
- 概述：研究残差连接/深层CNN，缓解深层网络退化并提升视觉识别性能。

## 966. Review of MRI-based Brain Tumor Image Segmentation Using Deep Learning Methods
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：综述论文
- 引用数：633
- 年份：2016
- 作者：Ali Işın, Cem Direkoğlu, Melike Şah
- 来源：Procedia Computer Science
- 链接：https://doi.org/10.1016/j.procs.2016.09.407
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 967. The early disease stage in axial spondylarthritis: Results from the german spondyloarthritis inception cohort
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：630
- 年份：2009
- 作者：Martin Rudwaleit, Hildrun Haibel, Xenofon Baraliakos et al.
- 来源：Arthritis &amp;amp; Rheumatism
- 链接：https://doi.org/10.1002/art.24483
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 968. DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：应用/方法论文
- 引用数：626
- 年份：2019
- 作者：Hanchao Li, Pengfei Xiong, Haoqiang Fan et al.
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00975
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 969. Review the state-of-the-art technologies of semantic segmentation based on deep learning
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：综述论文
- 引用数：625
- 年份：2022
- 作者：Yujian Mo, Yan Wu, Xinneng Yang et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2022.01.005
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 970. CondenseNet: An Efficient DenseNet Using Learned Group Convolutions
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：应用/方法论文
- 引用数：621
- 年份：2018
- 作者：Gao Huang, Shichen Liu, Laurens van der Maaten et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00291
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 971. Mobile-Former: Bridging MobileNet and Transformer
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, CNN
- 重要性：架构论文
- 引用数：617
- 年份：2022
- 作者：Yinpeng Chen, Xiyang Dai, Dongdong Chen et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00520
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 972. MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
- 类型：计算机视觉
- 标签：视觉, CNN, Segmentation
- 重要性：架构论文
- 引用数：614
- 年份：2019
- 作者：Yazan Abu Farha, Jürgen Gall
- 来源：2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr.2019.00369
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 973. Clinical course during the first 10 years of ulcerative colitis: results from a population-based inception cohort (IBSEN Study)
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：608
- 年份：2009
- 作者：Inger Camilla Solberg, Idar Lygren, Jørgen Jahnsen et al.
- 来源：Scandinavian Journal of Gastroenterology
- 链接：https://doi.org/10.1080/00365520802600961
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 974. Prediction of complicated disease course for children newly diagnosed with Crohn's disease: a multicentre inception cohort study
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：606
- 年份：2017
- 作者：Subra Kugathasan, Lee A Denson, Thomas D Walters et al.
- 来源：The Lancet
- 链接：https://doi.org/10.1016/s0140-6736(17)30317-3
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 975. Data augmentation using generative adversarial networks (CycleGAN) to improve generalizability in CT segmentation tasks
- 类型：计算机视觉 / 生成模型
- 标签：视觉, Generative AI, GAN, Segmentation
- 重要性：架构论文
- 引用数：605
- 年份：2019
- 作者：Veit Sandfort, Ke Yan, Perry J. Pickhardt et al.
- 来源：Scientific Reports
- 链接：https://doi.org/10.1038/s41598-019-52737-x
- 概述：研究生成对抗网络，通过生成器—判别器博弈学习数据分布。

## 976. Anatomically Constrained Neural Networks (ACNNs): Application to Cardiac Image Enhancement and Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：架构论文
- 引用数：604
- 年份：2018
- 作者：Ozan Oktay, Enzo Ferrante, Konstantinos Kamnitsas et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2017.2743464
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 977. CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers
- 类型：计算机视觉 / 系统/框架
- 标签：视觉, Framework/System, Attention/Transformer, Segmentation
- 重要性：系统/框架
- 引用数：600
- 年份：2023
- 作者：Jiaming Zhang, Huayao Liu, Kailun Yang et al.
- 来源：IEEE Transactions on Intelligent Transportation Systems
- 链接：https://doi.org/10.1109/tits.2023.3300537
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 978. DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：589
- 年份：2022
- 作者：Ailiang Lin, Bingzhi Chen, Jiayu Xu et al.
- 来源：IEEE Transactions on Instrumentation and Measurement
- 链接：https://doi.org/10.1109/tim.2022.3178991
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 979. Automatic Skin Lesion Segmentation Using Deep Fully Convolutional Networks With Jaccard Distance
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：586
- 年份：2017
- 作者：Yading Yuan, Ming Chao, Yeh-Chi Lo
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2017.2695227
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 980. Reliable Tuberculosis Detection Using Chest X-Ray With Deep Learning, Segmentation and Visualization
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：应用/方法论文
- 引用数：585
- 年份：2020
- 作者：Tawsifur Rahman, Amith Khandakar, Muhammad Abdul Kadir et al.
- 来源：IEEE Access
- 链接：https://doi.org/10.1109/access.2020.3031384
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 981. PU-Net: Point Cloud Upsampling Network
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：架构论文
- 引用数：584
- 年份：2018
- 作者：Lequan Yu, Xianzhi Li, Chi-Wing Fu et al.
- 来源：2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
- 链接：https://doi.org/10.1109/cvpr.2018.00295
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 982. Brain tumor segmentation based on deep learning and an attention mechanism using MRI multi-modalities brain images
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：应用/方法论文
- 引用数：581
- 年份：2021
- 作者：Ramin Ranjbarzadeh, Abbas Bagherian Kasgari, Saeid Jafarzadeh Ghoushchi et al.
- 来源：Scientific Reports
- 链接：https://doi.org/10.1038/s41598-021-90428-8
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 983. Asymmetric Non-Local Neural Networks for Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：架构论文
- 引用数：575
- 年份：2019
- 作者：Zhen Zhu, Mengdu Xu, Song Bai et al.
- 来源：2019 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2019.00068
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 984. A Brief Survey on Semantic Segmentation with Deep Learning
- 类型：计算机视觉
- 标签：视觉, Segmentation
- 重要性：综述论文
- 引用数：574
- 年份：2020
- 作者：Shijie Hao, Yuan Zhou, Yanrong Guo
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2019.11.118
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 985. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Reasoning/Test-time Scaling, Segmentation
- 重要性：架构论文
- 引用数：570
- 年份：2019
- 作者：Guotai Wang, Wenqi Li, Michael Aertsen et al.
- 来源：Neurocomputing
- 链接：https://doi.org/10.1016/j.neucom.2019.01.103
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 986. End-to-End Video Instance Segmentation with Transformers
- 类型：计算机视觉 / 视频/世界模型
- 标签：视觉, Video, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：567
- 年份：2021
- 作者：Yuqing Wang, Zhaoliang Xu, Xinlong Wang et al.
- 来源：2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr46437.2021.00863
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 987. The frequency and outcome of lupus nephritis: results from an international inception cohort study
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：561
- 年份：2016
- 作者：John G. Hanly, Aidan G. O’Keeffe, Li Su et al.
- 来源：Rheumatology
- 链接：https://doi.org/10.1093/rheumatology/kev311
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 988. Transfer learning using VGG-16 with Deep Convolutional Neural Network for Classifying Images
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：557
- 年份：2019
- 作者：Srikanth Tammina
- 来源：International Journal of Scientific and Research Publications (IJSRP)
- 链接：https://doi.org/10.29322/ijsrp.9.10.2019.p9420
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

## 989. FAT-Net: Feature adaptive transformers for automated skin lesion segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：557
- 年份：2022
- 作者：Huisi Wu, Shihuai Chen, Guilian Chen et al.
- 来源：Medical Image Analysis
- 链接：https://doi.org/10.1016/j.media.2021.102327
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 990. Object Detection via a Multi-region and Semantic Segmentation-Aware CNN Model
- 类型：计算机视觉
- 标签：视觉, CNN, Object Detection, Segmentation
- 重要性：架构论文
- 引用数：556
- 年份：2015
- 作者：Spyros Gidaris, Nikos Komodakis
- 来源：2015 IEEE International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2015.135
- 概述：研究深度目标检测框架，在定位与分类之间取得速度和精度平衡。

## 991. ReLayNet: retinal layer and fluid segmentation of macular optical coherence tomography using fully convolutional networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：554
- 年份：2017
- 作者：Abhijit Guha Roy, Sailesh Conjeti, Sri Phani Krishna Karri et al.
- 来源：Biomedical Optics Express
- 链接：https://doi.org/10.1364/boe.8.003627
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 992. Expectation-Maximization Attention Networks for Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：548
- 年份：2019
- 作者：Xia Li, Zhisheng Zhong, Jianlong Wu et al.
- 来源：2019 IEEE/CVF International Conference on Computer Vision (ICCV)
- 链接：https://doi.org/10.1109/iccv.2019.00926
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 993. Reducing the Hausdorff Distance in Medical Image Segmentation With Convolutional Neural Networks
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, Segmentation
- 重要性：架构论文
- 引用数：537
- 年份：2020
- 作者：Davood Karimi, Septimiu E. Salcudean
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2019.2930068
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 994. Swin Transformer Embedding UNet for Remote Sensing Image Semantic Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：535
- 年份：2022
- 作者：Xin He, Yong Zhou, Jiaqi Zhao et al.
- 来源：IEEE Transactions on Geoscience and Remote Sensing
- 链接：https://doi.org/10.1109/tgrs.2022.3144165
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 995. An effective CNN and Transformer complementary network for medical image segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, CNN, Segmentation
- 重要性：架构论文
- 引用数：534
- 年份：2023
- 作者：Feiniu Yuan, Zhengxiao Zhang, Zhijun Fang
- 来源：Pattern Recognition
- 链接：https://doi.org/10.1016/j.patcog.2022.109228
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 996. MISSFormer: An Effective Transformer for 2D Medical Image Segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：529
- 年份：2023
- 作者：Xiaohong Huang, Zhifang Deng, Dandan Li et al.
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2022.3230943
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 997. Stratified Transformer for 3D Point Cloud Segmentation
- 类型：计算机视觉
- 标签：视觉, Attention/Transformer, Segmentation
- 重要性：架构论文
- 引用数：528
- 年份：2022
- 作者：Xin Lai, Jianhui Liu, Li Jiang et al.
- 来源：2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52688.2022.00831
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 998. A review of the application of deep learning in medical image classification and segmentation
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, Segmentation
- 重要性：综述论文
- 引用数：521
- 年份：2020
- 作者：Lei Cai, Jingyang Gao, Di Zhao
- 来源：Annals of Translational Medicine
- 链接：https://doi.org/10.21037/atm.2020.02.44
- 概述：研究图像/语义分割网络，用像素级预测解决视觉理解问题。

## 999. Framing U-Net via Deep Convolutional Framelets: Application to Sparse-View CT
- 类型：计算机视觉 / 医疗/生命科学AI
- 标签：视觉, Medical AI, CNN, MoE/Sparse, Segmentation
- 重要性：架构论文
- 引用数：520
- 年份：2018
- 作者：Yoseob Han, Jong Chul Ye
- 来源：IEEE Transactions on Medical Imaging
- 链接：https://doi.org/10.1109/tmi.2018.2823768
- 概述：提出或扩展 U-Net 编解码与跳连结构，常用于医学图像和语义分割。

## 1000. Poly Kernel Inception Network for Remote Sensing Detection
- 类型：计算机视觉
- 标签：视觉, CNN
- 重要性：架构论文
- 引用数：518
- 年份：2024
- 作者：Xinhao Cai, Qiuxia Lai, Yuwei Wang et al.
- 来源：2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- 链接：https://doi.org/10.1109/cvpr52733.2024.02617
- 概述：研究卷积神经网络结构或训练方法，服务于图像分类、检测与视觉表征。

---

# 附录：2025–2026 全球新近重要但暂不纳入引用排序 Top 1000 的论文

说明：这些论文/技术报告发布时间较新，引用数尚未稳定；纳入标准是全球范围内公开影响较大、模型/代码/报告被社区广泛讨论，或提出了已显示有效的训练、推理、架构、多模态/世界模型方法。它们放在附录，不参与正文按引用数排序。

## A1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- 类型：NLP/语言模型 / 强化学习
- 标签：Language Model, RL, Reasoning/Test-time Scaling
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：DeepSeek-AI, Daya Guo, Dejian Yang et al.
- 来源：DeepSeek-AI / arXiv
- 链接：https://arxiv.org/abs/2501.12948
- 概述：系统展示通过大规模强化学习激发LLM推理能力，推动开源推理模型和“RL for reasoning”路线快速发展。

## A2. Kimi k1.5: Scaling Reinforcement Learning with LLMs
- 类型：NLP/语言模型 / 强化学习
- 标签：Language Model, RL
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Kimi Team, Angang Du, Bofei Gao et al.
- 来源：Moonshot AI / arXiv
- 链接：https://arxiv.org/abs/2501.12599
- 概述：总结Kimi k1.5的长链路强化学习和推理扩展经验，是2025年LLM推理训练路线的重要技术报告。

## A3. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- 类型：多模态
- 标签：Multimodal, Reasoning/Test-time Scaling
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Gemini Team / Google DeepMind et al.
- 来源：Google DeepMind / arXiv
- 链接：https://arxiv.org/abs/2507.06261
- 概述：Google DeepMind 的 Gemini 2.5 技术报告，覆盖推理、多模态、长上下文和智能体能力，是全球闭源前沿模型的重要公开资料。

## A4. AlphaEvolve: A coding agent for scientific and algorithmic discovery
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Alexander Novikov, Ngân Vũ, Marvin Eisenberger et al.
- 来源：Google DeepMind / arXiv
- 链接：https://arxiv.org/abs/2506.13131
- 概述：展示基于Gemini的编码智能体可用于算法发现和科学问题优化，是AI for science与自动化科研方向的重要案例。

## A5. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- 类型：视频/世界模型
- 标签：Video, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Meta AI Research et al.
- 来源：Meta AI / arXiv
- 链接：https://arxiv.org/abs/2506.09985
- 概述：Meta 的视频自监督世界模型，强调理解、预测和规划能力，延续JEPA路线并面向机器人/具身智能。

## A6. Gemma 3 Technical Report
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Gemma Team / Google DeepMind et al.
- 来源：Google / arXiv
- 链接：https://arxiv.org/abs/2503.19786
- 概述：Google 开源Gemma 3模型族技术报告，覆盖多模态、长上下文和高效部署，是2025年重要开放模型之一。

## A7. Titans: Learning to Memorize at Test Time
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Ali Behrouz, Peilin Zhong, Vahab Mirrokni et al.
- 来源：Google Research / arXiv
- 链接：https://arxiv.org/abs/2501.00663
- 概述：提出测试时记忆机制，让模型在推理阶段动态写入和检索长期记忆，属于长上下文/记忆增强架构的重要探索。

## A8. 2 OLMo 2 Furious
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Allen Institute for AI et al.
- 来源：Ai2 / arXiv
- 链接：https://arxiv.org/abs/2501.00656
- 概述：OLMo 2开放语言模型报告，公开训练配方、数据和模型，代表全球开放LLM透明化路线。

## A9. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
- 类型：深度学习相关
- 标签：Attention/Transformer, MoE/Sparse
- 重要性：架构论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：DeepSeek-AI et al.
- 来源：DeepSeek-AI / arXiv
- 链接：https://arxiv.org/abs/2502.11089
- 概述：提出面向硬件友好的可训练稀疏注意力机制，用于降低长上下文Transformer的计算和显存成本。

## A10. Qwen2.5-VL Technical Report
- 类型：多模态
- 标签：Multimodal
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Qwen Team et al.
- 来源：Alibaba / arXiv
- 链接：https://arxiv.org/abs/2502.13923
- 概述：Qwen视觉语言模型技术报告，覆盖图文理解、OCR、定位、视频理解等多模态能力。

## A11. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling
- 类型：多模态
- 标签：Multimodal
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：DeepSeek-AI et al.
- 来源：DeepSeek-AI / arXiv
- 链接：https://arxiv.org/abs/2501.17811
- 概述：统一多模态理解和生成的模型报告，通过数据与模型扩展提升图文理解和生成效果。

## A12. Qwen3 Technical Report
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Qwen Team et al.
- 来源：Alibaba / arXiv
- 链接：https://arxiv.org/abs/2505.09388
- 概述：Qwen3模型族技术报告，强调推理/非推理模式、MoE与多语言能力，是2025年重要开源LLM系列之一。

## A13. MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention
- 类型：深度学习相关
- 标签：Attention/Transformer, Reasoning/Test-time Scaling
- 重要性：训练/推理方法
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：MiniMax Team et al.
- 来源：MiniMax / arXiv
- 链接：https://arxiv.org/abs/2506.13585
- 概述：结合Lightning Attention与测试时计算扩展，探索更高效的长上下文和推理模型训练/推理方案。

## A14. Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning
- 类型：深度学习相关
- 标签：Attention/Transformer, MoE/Sparse, Reasoning/Test-time Scaling
- 重要性：模型/技术报告
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2026
- 作者：NVIDIA Research et al.
- 来源：NVIDIA / arXiv
- 链接：https://arxiv.org/abs/2606.15007
- 概述：NVIDIA开放MoE混合Mamba-Transformer模型报告，聚焦智能体推理、效率和长上下文能力。

## A15. Kimi Linear: An Expressive, Efficient Attention Architecture
- 类型：深度学习相关
- 标签：Attention/Transformer
- 重要性：架构论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Moonshot AI / Kimi Team et al.
- 来源：Moonshot AI / arXiv
- 链接：https://arxiv.org/abs/2510.26692
- 概述：提出高效线性注意力架构，目标是在长上下文场景中兼顾表达能力、训练稳定性和推理效率。

## A16. s1: Simple test-time scaling
- 类型：深度学习相关
- 标签：Reasoning/Test-time Scaling
- 重要性：训练/推理方法
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Niklas Muennighoff et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/2501.19393
- 概述：用小规模高质量推理数据和budget forcing复现测试时扩展效果，强调“简单方法也能显著提升推理”。

## A17. LIMO: Less is More for Reasoning
- 类型：深度学习相关
- 标签：Reasoning/Test-time Scaling
- 重要性：训练/推理方法
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：LIMO Team et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/2502.03387
- 概述：提出少量高质量样本即可激发基础模型数学推理能力的观点，对推理SFT数据效率讨论影响较大。

## A18. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- 类型：NLP/语言模型 / 强化学习 / 系统/框架
- 标签：Language Model, RL, Framework/System
- 重要性：系统/框架
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Qiying Yu, Zheng Zhang, Ruofei Zhu et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/2503.14476
- 概述：开源大规模LLM强化学习训练系统和关键技巧，面向复现推理模型RL训练流程。

## A19. DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition
- 类型：NLP/语言模型 / 强化学习
- 标签：Language Model, RL, Reasoning/Test-time Scaling
- 重要性：训练/推理方法
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：DeepSeek-AI et al.
- 来源：DeepSeek-AI / arXiv
- 链接：https://arxiv.org/abs/2504.21801
- 概述：面向形式化数学证明，结合子目标分解和强化学习提升LLM在Lean等证明任务上的能力。

## A20. Self-Adapting Language Models
- 类型：NLP/语言模型
- 标签：Language Model
- 重要性：应用/方法论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Research team et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/2506.10943
- 概述：研究语言模型在推理或使用过程中自适应更新/调整的机制，属于后训练和测试时学习方向的新近代表。

## A21. Simple Policy Gradients for Reasoning with Diffusion Language Models
- 类型：NLP/语言模型 / 生成模型 / 强化学习
- 标签：Language Model, Generative AI, RL, Optimization/Training, Diffusion/Flow, Reasoning/Test-time Scaling
- 重要性：训练/推理方法
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2025
- 作者：Research team et al.
- 来源：arXiv
- 链接：https://arxiv.org/abs/2510.04019
- 概述：把策略梯度用于扩散语言模型的推理能力提升，代表2025年“扩散式语言模型 + 强化学习”探索方向。

## A22. mHC: Manifold-Constrained Hyper-Connections
- 类型：深度学习相关
- 标签：Deep Learning
- 重要性：应用/方法论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2026
- 作者：Zhenda Xie, Yixuan Wei, Huanqi Cao et al.
- 来源：DeepSeek-AI / arXiv
- 链接：https://arxiv.org/abs/2512.24880
- 概述：针对 PreNorm 残差流随深度累积导致的表示稀释/信号放大问题，引入多残差流 Hyper-Connections，并用流形约束稳定跨层混合。

## A23. Attention Residuals
- 类型：深度学习相关
- 标签：Attention/Transformer, Residual/Skip Connection
- 重要性：奠基/方法论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2026
- 作者：Kimi Team / Moonshot AI
- 来源：MoonshotAI / arXiv
- 链接：https://arxiv.org/abs/2603.15031
- 概述：把标准残差的“逐层等权累加”改为对历史层输出做注意力聚合，使后层可选择性读取早期表征，缓解 PreNorm residual dilution。

## A24. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning
- 类型：视频/世界模型
- 标签：Video, Self-supervised Learning
- 重要性：应用/方法论文
- 引用数：新近论文，引用尚未稳定，不参与 Top 1000 引用排序
- 年份：2026
- 作者：Meta AI Research et al.
- 来源：Meta AI / arXiv
- 链接：https://arxiv.org/abs/2603.14482
- 概述：进一步改进V-JEPA视频自监督表征，强调更密集、更可迁移的视频特征，用于理解和规划。
