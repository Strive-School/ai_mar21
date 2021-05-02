# 20 Questions about Feature Engeneering

### 1. What are the train & validation sizes (num. of rows) when we apply crossVal with K=5 for a dataset of 100 rows?
- [ ] 95 rows for train, 5 for test (at each fold).
- [ ] 90 rows for train, 10 for test (at each fold).
- [x] 80 rows for train, 20 for test (at each fold).
- [ ] 75 rows for train, 25 for test (at each fold).

### 2. Imagine you have a medical dataset where each patient contains several rows. What is the best cross validation strategy?
- [ ] Cross validation (`KFold`)
- [ ] Stratified cross validation (`StratifiedKFold`)
- [x] Group cross validation (`GroupKFold`)
- [ ] Time cross validation (`TimeSeriesSplit`)

### 3. Imagine shop dataset and you want to predict the sales for the next month. What is the best cross validation strategy?
- [ ] Cross validation (`KFold`)
- [ ] Stratified cross validation (`StratifiedKFold`)
- [ ] Group cross validation (`GroupKFold`)
- [x] Time cross validation (`TimeSeriesSplit`)

### 4. When we use `OrdinalEncoder` for encoding categories?
- [ ] For Linear models
- [x] For Tree based models
- [ ] For SVMs and Neural Nets
- [ ] For KNNs

### 5. When can the `BinaryEncoder` be helpful?
- [ ] For numerical vars
- [x] For categorical vars with high cardinality
- [ ] For categorical vars which are very unbalanced
- [ ] For text vars

### 6. When can the `FrequencyEncoder` (aka `CountEncoder`) be helpful?
- [ ] For numerical vars
- [ ] For categorical vars with high cardinality
- [x] For categorical vars which are very unbalanced
- [ ] For time vars

### 7. When we use `OneHotEncoder` for encoding categories?
- [ ] For Linear models
- [ ] For Tree based models
- [ ] For SVMs and Neural Nets
- [x] For KNNs, Linear Models, SVMs and Neural Nets


### 8. For trees of the same depth, wich categorical encoder will achieve better resutls in most cases?
- [ ] OnHotEncoder
- [ ] OrdianalEncoder
- [ ] BinaryEncoder
- [x] TargetEncoder (aka MeanEncoder)

### 9. How we decice our models for our ensemble?
- [ ] Only pick our best tree-based models (XGBoost + LightGBM + Catboost)
- [ ] Only pick our best muliplicative models (Linear + SVM + NN)
- [x] Pick the best model of each family (Gradient Boosting + NN + KNN)
- [ ] Pick all our models (good and bad models)

### 10. What is best order for creating a pipeline for encode the variables?
- [ ] First encode the variables, the  fill the missings.
- [x] First fill the missings, the encode the variables.

### 11. How we create time based features (lags feats) with the target?
- [ ] We only encode information about the future.
- [x] We only encode information about the past.
- [ ] We encode information about the past and the future.
- [ ] We only encode information about the present.

### 12. When we create lags features, a common mistake is to introduce:
- [ ] Curse of dimesionality
- [x] Data leakage (aka leaks)
- [ ] Unbalanced data
- [ ] Entropy

### 13. What are good features to extract from the month?
- [ ] Number of the month [1,...,12]
- [ ] Days of the month [28,...,31]
- [ ] Number of weekend days (saturdays+sundays) [8,...,10]
- [x] All the previous features could be useful

### 14. What is the best integer type to encode the year for optimize the memory?
- [ ] Integer of 8 bits (`np.int8` or `np.uint8`)
- [x] Integer of 16 bits (`np.int16` or `np.uint16`)
- [ ] Integer of 32 bits (`np.int32` or `np.uint32`)
- [ ] Integer of 64 bits (`np.int64` or `np.uint64`)

## 15. What is a good library for encoding time data?
- [x] `tsfresh`
- [ ] `fuzzywuzzy`
- [ ] `seaborn`
- [ ] `matplotlib`

### 16. How we can encode the cities?
- [ ] We can extract coordintes (Lat & Lon)
- [ ] We can extract the region and the country.
- [ ] We can extract the wealth and economic wellness.
- [x] All the prevous options are possible.

### 17. When is useful rotating the map?
- [x] For Tree models
- [ ] For Linear models
- [ ] For Neural Nets
- [ ] For SVMs

### 18. What of these Feature Importance plots does not need to train a ML model?
- [x] Correlation of the input varibles with the target varible.
- [ ] XGBoost Feature Importance
- [ ] Permutation Feature Importance
- [ ] SHAP

### 19. What is a good library for an Automated Feature Engineering?
- [ ] Numpy
- [ ] Pandas
- [x] FeatureTools
- [ ] CatBoost

### 20. If you are on the top possition of the public leaderboard on kaggle, what that means?
- [ ] You are awesome, and you will win the competition for sure.
- [ ] Your solution is very good, and you should not team up with anyone, because you have the best model.
- [ ] You are Russian and you will win, because you speaks Russian and you are on the [ODS](https://ods.ai/) Slack chat.
- [x] You are doing well, but you could overfit the public LB and not winning.
