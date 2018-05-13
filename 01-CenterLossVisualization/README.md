# Center Loss Visualization on MNIST datasets

## Purpose: Observe the cluster influenced by center loss

一般做分類時是使用 Softmax + Crossentropy 來訓練模型的參數，但在某些情況下我們更關心透過 NN 是否能提取出好的 feature。可惜的是直接利用 Softmax + Crossentropy 來訓練模型所得到的 feature 並不一定會有 cluster 的效果。

仔細想想利用 Softmax + Crossentropy 來訓練模型只可以將每筆資料盡量歸到他所屬的類別，但卻不能使不同類別間的距離擴大，因此解由此方式訓練模型並不一定會得到 cluster 效果的 feature。

然而對 loss function 增加 center loss 的概念可以使每筆資料盡量歸到他所屬的類別並且擴大不同類別之間的距離，進而影響模型提取出具有 cluster 效果的 feature。

## Data 簡介

MNIST datasets 包含從零到九的手繪數字的灰度圖像。

每張圖像的高度為28像素，寬度為28像素，總共為784像素。 每個像素都有一個與之相關的像素值，表示該像素的亮度或暗度，較高的數字意味著較暗。 此像素值是一個介於0和255之間的整數，包括0和255。

* train.csv: 訓練數據集有785行，其中第一列是由用戶繪製的數字，也就是標籤。其餘列包含關聯圖像的像素值。

* test.csv: 跟 train.csv 是一樣的，只是缺少了第一行標籤的訊息。

## Summary

<div class="half">
    <img src="Output/NonCenter/epoch=0.jpg" height="300px">
    <img src="Output/IsCenter/epoch=0.jpg" height="300px">
    <img src="Output/NonCenter/epoch=29.jpg" height="300px">
    <img src="Output/IsCenter/epoch=29.jpg" height="300px">
</div>


## File Stucture

```
01-CenterLossVisualization
|    README.md
|    main.py
|    test.py
|
└─── Base
|      __init__.py
|      DataProcessing.py
|      Utility.py
|      Model.py
|      Train.py
|      Predict.py
|
└─── 01-RAWData
|       train.csv
|       test.csv
|
└─── 02-Output
|       ...
|___
```

## Reference

* [Keras中自定義復雜的loss函數](https://kexue.fm/archives/4493)

* [Keras callbacks guide and code](https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/)

* [【Technical Review】ECCV16 Center Loss及其在人臉識別中的應用](https://zhuanlan.zhihu.com/p/23340343)

* [MNIST center loss pytorch](https://github.com/jxgu1016/MNIST_center_loss_pytorch)