# Digit Recognizer

## Purpose: 


## Data 簡介

MNIST datasets 包含從零到九的手繪數字的灰度圖像。

每張圖像的高度為28像素，寬度為28像素，總共為784像素。 每個像素都有一個與之相關的像素值，表示該像素的亮度或暗度，較高的數字意味著較暗。 此像素值是一個介於0和255之間的整數，包括0和255。

* train.csv: 訓練數據集有785行，其中第一列是由用戶繪製的數字，也就是標籤。其餘列包含關聯圖像的像素值。

* test.csv: 跟 train.csv 是一樣的，只是缺少了第一行標籤的訊息。

## Summary


<div class="half">
    <img src="02-Output/NonCenter/lambda=0epoch=0.jpg" height="300px">
    <img src="02-Output/IsCenter/lambda=0.5epoch=0.jpg" height="300px">
    <img src="02-Output/NonCenter/lambda=0epoch=29.jpg" height="300px">
    <img src="02-Output/IsCenter/lambda=0.5epoch=29.jpg" height="300px">
</div>


## File Stucture

```
02-Kaggle-DigitRecognizer
|    README.md
|    main.py
|    Test.py
|
└─── Base
|      __init__.py
|      DataProcessing.py
|      Predict.py
|      Utility.py
|      Model.py
|      Train.py
|
└─── 01-RAWData
|       train.csv
|       test.csv
|       sample_submission.csv
|
└─── 02-Output
|       submission.csv
|       01Train
|         log.csv
|         LossAccuracyCurves.png
|         model.h5
|
|       02Test
|         log.csv
|         LossAccuracyCurves.png
|         model.h5
|___
```

## Reference

* [Keras中自定義復雜的loss函數](https://kexue.fm/archives/4493)

* [Keras callbacks guide and code](https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/)

* [【Technical Review】ECCV16 Center Loss及其在人臉識別中的應用](https://zhuanlan.zhihu.com/p/23340343)

* [MNIST center loss pytorch](https://github.com/jxgu1016/MNIST_center_loss_pytorch)