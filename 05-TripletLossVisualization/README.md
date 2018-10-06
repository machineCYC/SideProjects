# Triplet Loss Visualization on MNIST datasets

## Purpose: Observe the cluster influenced by triplet loss

一般做圖片多類別分類的常見方法為，建立一個捲積網絡，Softmax 當作輸出，Crossentropy 當作 loss function 來訓練模型的參數。而這種方法有個缺點就是當要新增一個類別的時候，必須要重新訓練模型。最常見的例子就是人臉辨識系統，資料庫中的使用者會一直改變 (人員會增加或減少)，採用此方式在人員增加或減少時就會造成許多困擾。

由此可知這種訓練方式，模型的擴充能力相當不足。因此衍生出 embedding 的概念，及將圖片 mapping 到歐式空間中的向量，最後在藉此向量來衡量彼此是否相似。這種方式仰賴的 loss function 跟一般認知的不太相同。而是一種 triplet loss。

所以希望藉由這次的實驗，可以了解 triplet loss 的概念和訓練流程。

## Data 簡介

MNIST datasets 包含從零到九的手繪數字的灰度圖像。

每張圖像的高度為28像素，寬度為 28 像素，總共為 784 像素。 每個像素都有一個與之相關的像素值，表示該像素的亮度或暗度，較高的數字意味著較暗。 此像素值是一個介於 0 和 255 之間的整數，包括 0 和 255。

* training set: 55000 筆
* validation set: 5000 筆
* testing set: 10000 筆

## Summary


## Note


## File Stucture

```
05-TripletLossVisualization/
|    .gitignore
|    README.md
|    train.py
|    visulization.py
|
└─── experiments/
|      naive_cnn
|
|
└─── src/
|      __init__.py
|      utils.py
|      create_metadata_tsv.py
|      create_sprite_image.py
|
└─── model/
|      __init__.py
|      triplet_loss.py
|      model_fn.py
|
|
└─── data/
|      MNIST_data/
|       
|___
```

# Reference

* [Facenet](https://blog.csdn.net/qq_15192373/article/details/78490726)

* [Triplet Loss Example(keras)](https://github.com/SpikeKing/triplet-loss-mnist)

* [Triplet blog](https://omoindrot.github.io/triplet-loss)

* [Tensorflow Estimator](https://zhuanlan.zhihu.com/p/33681224)

* [Tensorboard Embedding](https://easytired.wordpress.com/2018/02/01/tensorboard-embedding/)