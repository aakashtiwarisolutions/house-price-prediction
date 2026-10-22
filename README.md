# House Price Prediction

Predicting house sale prices from 79 features. I compared linear regression, Ridge, KNN, random forest, gradient boosting and XGBoost, tried feature engineering, and blended Ridge with XGBoost for the final Kaggle submission.

## Data

- train.csv: 1460 houses with their sale price
- test.csv: 1459 houses without the sale price, used for the Kaggle submission
- data_description.txt: what each column means

## Steps

1. Load the data and look at the sale price distribution
2. Check missing values. Most were not really missing. For example, an empty PoolQC means the house has no pool, so these were filled with "None"
3. Fill the remaining gaps: MasVnrArea with 0, GarageYrBlt with the year the house was built, LotFrontage with the median and Electrical with the most common value
4. Find the features most related to price using correlation
5. Remove 2 outliers: very large houses (over 4000 sq ft) that sold for a low price
6. Take the log of the sale price, because a few expensive houses made the prices very skewed
7. Drop Id and one-hot encode all 43 text columns (79 features became about 290 columns)
8. Compare models with 5-fold cross-validation
9. Try new features and check if they improve the models
10. Blend Ridge and XGBoost
11. Prepare the test set the same way, align its columns with the training data and predict

## Top features by correlation with price

| Feature | Meaning | Correlation |
|---------|---------|-------------|
| TotalSF | Basement + 1st floor + 2nd floor (added feature) | 0.83 |
| OverallQual | Overall quality rating (1 to 10) | 0.80 |
| GrLivArea | Living area above ground | 0.73 |
| TotalBsmtSF | Basement size | 0.65 |
| 1stFlrSF | First floor size | 0.63 |

## Model comparison

5-fold cross-validation, RMSE on log price (lower is better):

| Model | RMSE |
|-------|------|
| XGBoost (500 trees) | 0.118 |
| Gradient boosting (300 trees) | 0.120 |
| Ridge (alpha=10, scaled) | 0.126 |
| Linear regression | 0.128 |
| Random forest (300 trees) | 0.138 |
| KNN (k=10, scaled) | 0.196 |

- XGBoost gave the lowest error.
- The random forest was weaker, likely because averaging many trees makes it less accurate for very cheap and very expensive houses.
- KNN was the weakest. It treats all 320 columns as equally important when finding similar houses. Using only 8 important features improved it from 0.196 to 0.165, but it still stayed behind the other models.
- Scaling mattered for KNN: without it the error was 0.226.

## Feature engineering

New features added: TotalSF, TotalBath, HouseAge, RemodAge, TotalPorch, OverallQual × TotalSF, and MSSubClass changed from a number to a category.

| Features | Ridge | XGBoost |
|----------|-------|---------|
| Original | 0.1257 | 0.1180 |
| With new features | 0.1268 | 0.1176 |

TotalSF became the feature most related to price, but the models did not improve in a meaningful way. XGBoost can already combine the floor areas by itself, and for Ridge TotalSF is just a sum of columns it already has.

## Blending

Predictions from Ridge and XGBoost were mixed with different weights:

| Ridge share | RMSE |
|-------------|------|
| 0% (XGBoost only) | 0.1182 |
| 20% | 0.1141 |
| 40% | 0.1128 |
| 50% | 0.1133 |
| 70% | 0.1165 |
| 100% (Ridge only) | 0.1261 |

40% Ridge and 60% XGBoost gave the best result, better than either model alone. The two models make different kinds of mistakes, so averaging them cancels part of the error.

## Kaggle results

| Submission | Score |
|------------|-------|
| XGBoost | 0.1325 |
| Blend: 40% Ridge + 60% XGBoost | 0.1319 |

## Files

- house_prices.ipynb: the notebook
- train.csv, test.csv, data_description.txt: Kaggle data
- submission.csv: XGBoost predictions
- submission_blend.csv: blended predictions
- requirements.txt

## Run

```
pip install -r requirements.txt
jupyter notebook house_prices.ipynb
```

## Links

- Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
