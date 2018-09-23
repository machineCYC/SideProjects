# Transfer Learning

## Purpose: Fine-tuning the last layer with pretrained model

Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer, then treat the rest of the ConvNet as a fixed feature extractor for the new dataset.

## Data 簡介


## Summary


## Note

tensorboard --logdir=tmp/retrain_logs/train/, tmp/retrain_logs/validation/

## File Stucture

```
04-TransferLearning/
|    README.md
|    main.py
|
└─── src/
|      __init__.py
|      file.py
|      model.py
|      process.py
|
└─── tmp/
|       imagenet/
|       retrain_logs/
|
|
└─── data/
|       train/
|       train2/
|       test/
|       train_img/
|       test_img
|___
```

# Reference

* [Tensorflow Graph](https://zhuanlan.zhihu.com/p/31308381)

* [Transfer Learning: retraining Inception V3 for custom image classification](https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557)