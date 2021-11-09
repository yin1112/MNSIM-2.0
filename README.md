# MNSIM_Python
**Citation Information**: Zhenhua Zhu, Hanbo Sun, Kaizhong Qiu, Lixue Xia, Gokul Krishnan, Guohao Dai, Dimin Niu, Xiaoming Chen, X. Sharon Hu, Yu Cao, Yuan Xie, Yu Wang, Huazhong Yang,  MNSIM 2.0: A Behavior-Level Modeling Tool for Memristor-based Neuromorphic Computing Systems, in Great Lakes Symposium on VLSI (GLSVLSI), 2020.

Zhenhua Zhu<sup>1*</sup>, Hanbo Sun<sup>1</sup>, Kaizhong Qiu<sup>1</sup>, Lixue Xia<sup>2</sup>, Gokul Krishnan<sup>6</sup>, Dimin Niu<sup>2</sup>, Qiuwen Lou<sup>3</sup>,

Xiaoming Chen<sup>4</sup>, Yuan Xie<sup>2, 5</sup>, Yu Cao<sup>6</sup>, X. Sharon Hu<sup>3</sup>, Yu Wang<sup>1*</sup>, and Huazhong Yang<sup>1</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>Alibaba Group, <sup>3</sup>University of Notre Dame, 
<sup>4</sup>Institute of Computing Technology, Chinese Academy of Sciences, 
<sup>5</sup>University of California, Santa Barbara,
<sup>6</sup>Arizona State University

<sup>*</sup>zhuzhenh18@mails.tsinghua.edu.cn, yu-wang@tsinghua.edu.cn

MNSIM_Python version 1.0 is still a beta version. If you have any questions and suggestions about MNSIM_Python please contact us via e-mail. We hope that MNSIM_Python can be helpful to your research work, and sincerely invite every PIM researcher to add your ideas to MNSIM_Python to enlarge its function.

For more information about MNSIM_Python, please refer to the MNSIM_manual.pdf

#Our work
The original project only supports the image classification function. On this basis, we have added the image classification function (the performance calculated by this function is not accurate).

We found that the author's quantization strategy is not suitable for image classification, so we chose other strategies. (from identity relu to clipped relu)

#How to train target detection network
- python3 synthesize.py -g 0 -d voc2007 -n yolo -t detection_train -m train 
