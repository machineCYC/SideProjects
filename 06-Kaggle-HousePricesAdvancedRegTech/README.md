# House Prices: Advanced Regression Techniques

## Purpose: House Prices Predict

每一間房子透過 79 個特正來描述，希望透過這些特徵來對房子做定價

## Data 簡介

- train.csv: 訓練資料，包含 1460 筆資料, 81 個欄位
- test.csv - 測試資料，包含 1459 筆資料, 80 個欄位
- data_description.txt: 完整的資料描述，包含每個欄位說明。
- sample_submission.csv: 一個 submit 的標準格式，包含資料 id 和銷售金額

## Summary

| Entries | train cv rmse | train rmse(註1) | ranking | total team | submit score(註2) |
| --- | --- | --- | --- |--- |
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