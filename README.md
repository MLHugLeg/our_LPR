# our_LPR
our implementation of LPR

## 写在前面 -- 叶剑南

我们只要在EasyPR和HyperLPR的框架下,适当使用OpenALPR的云服务，就可以进行机器学习课程的开发工作：

车牌识别的工作可以分为3步进行：
1. 车牌的检测与定位：这一步时比较复杂的一步，包含一些图像处理和openCV的使用的内容。也是我们可以重点研究的部分。

	即含车牌的bounding box的检测： 原库中使用的是opencv中SVM进行检测，我们可以考虑沿用SVM，并做一些改进。或者使用神经网络方法：如faster-RNN或者YOLO。

	原作者的博客写了很多关于这部分的处理（博客写的很好，我们可以借鉴很多）：https://blog.csdn.net/liuuze5/article/details/46290455

	我们也可以再看看HyperLPR中的定位方法: https://blog.csdn.net/relocy/article/details/78705662 他们号称能实现端到端的检测，且封装成了python接口，比较好调用。我们也比较容易引入其他框架做改进。

2. 字符的分割：即识别出车牌后如何将一个个字符分割出来.

	基本使用的是二值化后连通域提取的技术。在这篇博客中有介绍：http://www.cnblogs.com/subconscious/p/4660952.html

	我们可以使用其他方法进行改进，如这篇https://github.com/Bob-Yeah/HyperLPR-Training 介绍的。

3. 分割好的字符的分类：这一步就比较简单了，分割得到的字符只有地名简称，A-Z英文和0-9的数字，可以沿用LeNet的CNN网络进行训练。

	原库中使用的是ANN，可以比较容易的做出改进。

这里我觉的我们把简单的CNN引入即可。
