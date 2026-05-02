# Energy use and CO2 emissions based on 2016 Seattle data
Our task was to predict CO2 emissions and energy use for non-residential buildings by taking into account only structural variables of 2016 Seattle dataset. 
Targets : SiteEnergyUse(kBtu), TotalGHGEmissions
Plan of action :
1. Data cleaning.
Nulls, redundant data, not structural, derived from targets, impossible, outliers.
2. Feature engineering.
One-hot and GFA-weighted encoding, create 3 new features, delete redundant or having too many distint values columns, trim the distribution of a couple of top and bottom percentiles of target intensities (targets / surface) to eliminate outliers. 
3. Train the models.
Split 80/20 Train/test, kfold = 10, random_state = 42, and test multiple models:

    a. **Linear**
   
- Linear: simple linear model without regularization. Fast and interpretable (explainable/ easy to understand how each feature affects the predictions thanks to coefficients) but prone to overfitting with many features (our case).
- Ridge: linear regression with L2 regularization.Hadles multicollinearity well by shrinking correlated feature coefficients.
- Lasso: linear regression with L1 regularization. Automatically selects features by pushing less important coefficients to zero.
- ElasticNet: L1 + L2 regularization. Best of both worlds - handles multicollinearity and performs feature selection.
   
    b. **Tree-Based**
   
        - Random Forest: ensemble of decision trees trained on random subsets of data and features. Reduces overfitting through averaging.
        - XGBoost: gradient boosted trees that build models sequentially, each correcting previous errors. Often performs great in practice. (in our case ir probably doesn't have enough samples).
   
    c. **Support Vector Machine**
   
        - SVR (support vector regression): finds the optimal hyperplane that best fits the data within a margin of tolerance. Good for high-dimensional spaces. It's an alternative non-linear approach using kernel methods, useful for comparison.
   
    d. **Baseline**
   
        - Dummy Regressor: always predicts the mean of the training data. It's the minimum acceptable performance - any model must beat this. If our models don't significantly outperform this baseline, they're not adding value.
We observed overfitting, and clean datasets outperformed the ones with outliers.

Metrics:
+ R² (R-squared): proportion of the target's variance that the model explains. Maximize (closer to 1 = better).
+ MAE (Mean Absolute Error): average absolute difference between predictions and actual values (in the original units of the target). Minimize (closer to 0 = better). For interpretability. Robust to outliers.
+ RMSE (Root Mean Squared Error): identifies models that fail badly on extreme cases. Minimize (closer to 0 = better). Large errors are penalized more heavily. Sensitive to outliers.

Best performing models were:
Energy - ElasticNet
R2 = 0.729

CO2 - Lasso
R2 = 0.629
MAE = 75

4. Check if energystar is useful.
We impute a lot of data (thirs is nulls) for only a 1-2% performance gain, and this is just a performance evaluation. Since we should only use structural data, it's better to omit it.
5. Optimization and top features.
Top features are surface, top three types by surface are laboratory, medical office, and hotel in different order for the targets.
ElasticNet (Energy) best :
CV R² = 0.727
Test R² = 0.673

Lasso (CO2) best :
CV R² = 0.575
Test R² = 0.564

Good results for energy, ok results for CO2 but acceptable since it's a real messy data.

We tested the logarithmic transformation of GHG, but it significantly degraded performance (R² dropped from 0.564 to -266), so we retained the original scale.

## Conclusion.
This analysis demonstrates that structural variables alone can effectively predict building energy use and CO₂ emissions, with ElasticNet (R²=0.673) and Lasso (R²=0.564) delivering robust results. While energy predictions are strong, CO₂ emissions are noisier and achieve acceptable accuracy. 
The study highlights how important is data cleaning, feature selection, and model regularization in handling real-world datasets. Future work could explore non-linear models or additional structural features to further improve performance.
