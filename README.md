# House Price Prediction

Predicting house sale prices from 79 features using linear regression, Ridge, random forest, gradient boosting and XGBoost. The final model was submitted to the Kaggle House Prices competition.

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
8. Compare 5 models with 5-fold cross-validation
9. Train the best model on all data, prepare the test set the same way and predict

## Top features by correlation with price

| Feature | Meaning | Correlation |
|---------|---------|-------------|
| OverallQual | Overall quality rating (1 to 10) | 0.79 |
| GrLivArea | Living area above ground | 0.71 |
| GarageCars | Garage size in cars | 0.64 |
| GarageArea | Garage size in sq ft | 0.62 |
| TotalBsmtSF | Basement size | 0.61 |

## Results

5-fold cross-validation, RMSE on log price (lower is better):

| Model | RMSE |
|-------|------|
| Linear regression | 0.128 |
| Ridge (alpha=10, scaled) | 0.126 |
| Random forest (300 trees) | 0.138 |
| Gradient boosting (300 trees) | 0.120 |
| XGBoost (500 trees) | 0.118 |

XGBoost gave the lowest error. The random forest did worst, likely because averaging many trees makes it less accurate for very cheap and very expensive houses.

Kaggle score with XGBoost: 0.1325

## Files

- house_prices.ipynb: the notebook
- train.csv, test.csv, data_description.txt: Kaggle data
- submission.csv: predictions submitted to Kaggle
- requirements.txt

## Run

```
pip install -r requirements.txt
jupyter notebook house_prices.ipynb
```

## Links

- Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
