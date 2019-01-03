# House Prices: Advanced Regression Techniques

## Purpose: House Prices Predict

The goal is to predict the sales price for each house. Each house has 79 features to describe the building.

For each house Id in the test set, you must predict the value of the SalePrice variable.

Submissions are evaluated on [Root-Mean-Squared-Error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

## Data 簡介

- train.csv: Training data, including 1460 datas, 81 features
- test.csv - Testing data, including 1459 data, 80 features
- data_description.txt: Describe the detail for each house, including features introduction.
- sample_submission.csv: A standard submission file format, including house id and SalePrice.

## Summary

| Entries | train cv rmse | train rmse(註1) | ranking | total team | submit score(註2) |
| --- | --- | --- | --- |--- |--- |
| 1 | NA | NA | 4438 | 4691 | 0.40890 |
| 2 | NA | 0.08155 | 1161 | 4753 | 0.12210 |
| 3 | NA | 0.07636 | 978 | 4753 | 0.12042 |
| 4 | NA | 0.07572 | 973 | 4757 | 0.12039 |
| 4 | 0.1075 | 0.07599 | 424 | 4587 | 0.11564 |

註1:是透過 log(1+x) 轉換所計算出來的 score

註2:透過 exp(predict) - 1 轉換所計算出來的 score

1. 如果只針對 SalePrice 做 log-transform，不針對其他數值變數做 transform 結果會很差
2. 對 SalePrice 做 log-transform 和數值變數做 box-cox transform，最後利用 avg model，lambda=0.15，結果為 Entries2
3. 對 SalePrice 做 log-transform 和數值變數做 box-cox transform，最後利用 stacking model，lambda=0.15，結果為 Entries3
4. 對 SalePrice 做 log-transform 和數值變數做 box-cox transform，最後利用 stacking model，lambda=0.3，結果為 Entries4
5. 對 SalePrice 做 log-transform 和數值變數做 box-cox transform，最後利用 stacking model，lambda=-2,2 optim，結果為 Entries5

## Note

1. Log-Transform、SquareRoot-Transform: Skew must be positive
2. Exponential-Transforma: Skew must be negative


## File Stucture

```
06-Kaggle-HousePricesAdvancedRegTech
|    README.md
|    main.ipynb
|    eda.ipynb
|    requirements-to-freeze.txt
|
└─── data
|      train.csv
|      test.csv
|      sample_submission.csv
|      data_description.txt
|      submission.csv
|___
```

## Reference

* [Kaggle Competitions](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

* [如何在 Kaggle 首戰中進入前 10%](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/)

* [Tuning parameter in Gradient Boosting (GBM)](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)