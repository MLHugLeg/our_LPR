# our_LPR
our implementation of LPR

## 写在前面 -- 叶剑南

我们只要在EasyPR和HyperLPR的框架下,适当使用OpenALPR的云服务，就可以进行机器学习课程的开发工作：

车牌识别的工作可以分为3步进行：
1. 车牌的检测与定位：这一步时比较复杂的一步，包含一些图像处理和openCV的使用的内容。也是我们可以重点研究的部分。

	其他需要考虑的问题：倾斜校正

	即含车牌的bounding box的检测： 原库中使用的是opencv中SVM进行检测，我们可以考虑沿用SVM，并做一些改进。或者使用神经网络方法：如faster-RNN或者YOLO。

	原作者的博客写了很多关于这部分的处理（博客写的很好，我们可以借鉴很多）：https://blog.csdn.net/liuuze5/article/details/46290455

	我们也可以再看看HyperLPR中的定位方法: https://blog.csdn.net/relocy/article/details/78705662 他们号称能实现端到端的检测，且封装成了python接口，比较好调用。我们也比较容易引入其他框架做改进。

2. 字符的分割：即识别出车牌后如何将一个个字符分割出来.

	基本使用的是二值化后连通域提取的技术。在这篇博客中有介绍：http://www.cnblogs.com/subconscious/p/4660952.html

	我们可以使用其他方法进行改进，如这篇https://github.com/Bob-Yeah/HyperLPR-Training 介绍的。

3. 分割好的字符的分类：这一步就比较简单了，分割得到的字符只有地名简称，A-Z英文和0-9的数字，可以沿用LeNet的CNN网络进行训练。

	原库中使用的是ANN，可以比较容易的做出改进。

    这里我觉的我们把简单的CNN引入即可。

## 补充材料

### Andrew Ng 的神经网络课程有一章时目标检测。
可参考中文笔记的相关篇章，可以对整个过程有个大致的理解
https://github.com/fengdu78/deeplearning_ai_books/blob/master/Deeplearning%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0v5.47.pdf


## 数据集 Dataset
目前已经有的数据集是
### LPR/resources/train/下的压缩包。有车牌的样本，分割后的字符等。正负样本可以用来训练SVM或者haar检测器，分割后的含单个字符的文件可以用来训练字符分类器。
### 本仓库dataset中的dataset2.rar： 图片文件名就是车牌信息。
### 其他一些杂乱的数据集，可以各取所需。


## 新的技术路线

robust license plate detection and segmentation

end-to-end recognition: https://github.com/szad670401/end-to-end-for-chinese-plate-recognition

## 分工
### 从原始图片中分割出车牌区域（四个角点，并做好畸变矫正）
### 分割出来质量较好的图片的OCR识别（把分割和单个识别可以结合在一起）