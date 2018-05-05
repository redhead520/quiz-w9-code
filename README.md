# 第10周作业描述

原作业题目地址https://gitee.com/ai100/quiz-w9-code



### 代码目录结构：
```
├── convert_fcn_dataset.py
├── convert_fcn_dataset.sh
├── data(文件体积太大,超过100M,已经提交到.gitignore了)
│   ├── fcn_train.record
│   ├── fcn_val.record
│   └── vgg_16.ckpt
├── output
├── README.md
├── .gitignore
├── requirements.txt
├── dataset.py
├── train.py
├── train.sh
├── utils.py
├── val_1000_annotation.jpg
├── val_1000_img.jpg
├── val_1000_prediction_crfed.jpg
├── val_1000_prediction.jpg
└── vgg.py

```


### 数据集
- 生成的fcn_train.record,fcn_val.record与tinymind AI100上的数据集,大小上有0.几兆的区别,训练时未发现异常.
- 数据集地址:https://www.tinymind.com/jxhuanghf/datasets/quiz-fcn
![结果图](result_dataset.jpg)

### 模型训练完成：
训练结果
![结果图](result_running.jpg)


### 模型验证结果：
训练完成之后，在**/output/eval**下面生成验证图片

![结果图](result_eval_out.jpg)

原图 

![结果图](result_val_1000_img.jpg)

标签

![结果图](val_1000_annotation.jpg)

预测

![预测](result_val_1000_prediction.jpg)

CRF之后的预测
![预测](result_val_1000_prediction_crfed.jpg)


### 心得体会：
提供一份文档，描述自己的8Xfcn实现，需要有对关键代码的解释。描述自己对fcn的理解。


