# House Prices: Advanced Regression Techniques

## Purpose: House Prices Predict

每一間房子透過 79 個特正來描述，希望透過這些特徵來對房子做定價

## Data 簡介

- train.csv: 訓練資料，包含 1460 筆資料, 81 個欄位
- test.csv - 測試資料，包含 1459 筆資料, 80 個欄位
- data_description.txt: 完整的資料描述，包含每個欄位說明。
- sample_submission.csv: 一個 submit 的標準格式，包含資料 id 和銷售金額

## Summary

| Entries | train score(註1) | ranking | total team | submit score(註2) |
| --- | --- | --- | --- |--- |
| 1 | NA | 4438 | 4691 | 0.40890 |
| 2 | 0.08155 | 1161 | 4755 | 0.12210 |
| 3 | 0.07636 | 979 | 4760 | 0.12042 |

註1:是透過 log(1+x) 轉換所計算出來的 score

註2:透過 exp(predict) - 1 轉換所計算出來的 score

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
