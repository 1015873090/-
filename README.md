# Facial-Expression-Recognition
本项目针对面部表情识别任务，在 FER2013 数据集上系统性地探索了从传统机器学习到深度学习的解决方案。在传统方法方面，设计并评估了六种特征组合方案，重点分析了手工特征（HOG, LBP）与分类器（SVM, 随机森林）的组合效能，发现多特征融合策略显著提升了识别性能；在深度学习方面，构建了 VGG19 与 ResNet 基线模型，并引入眼部空间注意力机制以增强对判别性区域的特征提取能力。实验结果表明，两种方法各有优势，且基于空间注意力机制的模型显著改善了对关键部位的关注和整体分类性能。
本项目是我的本科毕业设计，在这里只提供核心源代码，实际运行需要您自己配置环境（如Python环境、FER2013数据集的下载等）。如果您需要完整的运行文件，请联系：1015873090@qq.com。
关于代码的描述：
main.py是训练代码，demo.py是使用训练出的模型进行人脸识别，model.py是具体模型配置，也是最体现工作量的部分，demo2.py是使用FER2013的PrivateTest集进行对模型的评估。
使用方法：
首先配置环境（这里对您可能最麻烦）。
运行main.py即进入训练模式开始训练，等着就行了。
运行demo.py即进入人脸识别模式（传统机器学习（HOG、LBP、SVM、RF）可能需要按空格键来进行人脸识别，按Q键退出）。
运行demo2.py，即对模型进行评估，等着就行了。
model.py是对训练的调整，要改但看不懂可以问AI。

This project systematically explores solutions from traditional machine learning to deep learning for facial expression recognition tasks on the FER2013 dataset. In terms of traditional methods, six feature combination schemes were designed and evaluated, with a focus on analyzing the combination efficiency of manual features (HOG, LBP) and classifiers (SVM, random forest). It was found that the multi feature fusion strategy significantly improved recognition performance; In terms of deep learning, VGG19 and ResNet baseline models were constructed, and an eye spatial attention mechanism was introduced to enhance the feature extraction ability for discriminative regions. The experimental results show that both methods have their own advantages, and the model based on spatial attention mechanism significantly improves the attention to key parts and overall classification performance.
This project is my undergraduate graduation project, and only the core source code is provided here. The actual operation requires you to configure the environment yourself (such as Python environment, download of FER2013 dataset, etc.). If you need the complete running file, please contact: 1015873090@qq.com .
About the code description: main.by is the training code, demo.by uses the trained model for face recognition, model. py is the specific model configuration and the most labor-intensive part, and demo2.py uses FER2013's Private Test set to evaluate the model.
Usage: 
First, configure the environment (which may be the most troublesome for you). Run main.py to enter training mode and start training, just wait.
Running demo. py will enter face recognition mode (traditional machine learning (HOG, LBP, SVM, RF) may require pressing the spacebar for face recognition, pressing the Q key to exit). 
Run demo2.py to evaluate the model, just wait. 
Model. py is an adjustment for training. If you want to make changes but don't understand, you can ask AI.
