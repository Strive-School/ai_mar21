<h1 align="center">Day 2: Data Cleaning (Missings and Outliers)</h1>

## Exercises

### ❓ Missing values

1. What is the missing datatype used in pandas?
2. How to replace all occurences of the value 9999 to missing in pandas?
3. How to get the absolute number of missings for each variable in pandas?
4. How to get the percentage of missings for each variable in pandas?
5. How to drop rows with missing values?
6. How to drop variables with missing values?
7. What is the univariate imputation method in sklearn?
8. What is the multivariate imputation method in sklearn?
9. What is the best univariate imputation method to categorical variables? (Explain why)
10. What is the best univariate imputation method to numerical variables? (Explain why)


### 🔎 Outliers

1. What is an outlier?
2. What is a simple method to detect and deal with outliers of a numerical variable?
3. What is novelty detection?
4. Name 4 advanced methods of outlier detection in sklearn.


### 🖋 Typos

1. What is a typo?
2. What is a good method of automatically detect typos?



### Practical case

Consider the following dataset: [San Francisco Building Permits](https://www.kaggle.com/aparnashastry/building-permit-applications-data). Look at the columns "Street Number Suffix" and "Zipcode". Both of these contain missing values.

- Which, if either, are missing because they don't exist?
- Which, if either, are missing because they weren't recorded?

Hint: Do all addresses generally have a street number suffix? Do all addresses generally have a zipcode?



| Var # |  NaN % | Var name                               | Var Description                                    |
|------:|-------:|:---------------------------------------|:---------------------------------------------------|
|     1 |      0 | Permit Number                          | Number assigned while filing                       |
|     2 |      0 | Permit Type                            | Type of the permit represented numerically.        |
|     3 |      0 | Permit Type Definition    | Description of the Permit type, for example new construction, alterations |
|     4 |      0 | Permit Creation Date      | Date on which permit created, later than or same as filing date           |
|     5 |      0 | Block                                  | Related to address                                 |
|     6 |      0 | Lot                                    | Related to address                                 |
|     7 |      0 | Street Number                          | Related to address                                 |
|     8 | 98.885 | **Street Number Suffix**               | Related to address                                 |
|     9 |      0 | Street Name                            | Related to address                                 |
|    10 |  1.391 | Street Name Suffix                     | Related to address                                 |
|    11 | 85.178 | Unit                                   | Unit of a building                                 |
|    12 | 99.014 | Unit suffix                            | Suffix if any, for the unit                        |
|    13 |  0.145 | Description         | Details about purpose of the permit. Example: reroofing, bathroom renovation     |
|    14 |      0 | Current Status                         | Current status of the permit application.          |
|    15 |      0 | Current Status Date                    | Date at which current status was entered           |
|    16 |      0 | Filed Date                             | Filed date for the permit                          |
|    17 |  7.511 | Issued Date                            | Issued date for the permit                         |
|    18 | 51.135 | Completed Date  | The date on which project was completed, applicable if Current Status = “completed”   |
|    19 |  7.514 | First Construction Document Date       | Date on which construction was documented          |
|    20 | 96.519 | Structural Notification                | Notification to meet some legal need, given or not |
|    21 | 21.510 | Number of Existing Stories | Num of existing stories in the building. Not applicable for certain permit types|
|    22 | 21.552 | Number of Proposed Stories             | Number of proposed stories for the construction/alteration    |
|    23 | 99.982 | Voluntary Soft-Story Retrofit          | Soft story to meet earth quake regulations      |
|    24 | 90.534 | Fire Only Permit                       | Fire hazard prevention related permit           |
|    25 | 26.083 | Permit Expiration Date                 | Expiration date related to issued permit.       |
|    26 | 19.138 | Estimated Cost                         | Initial estimation of the cost of the project   |
|    27 |  3.049 | Revised Cost                           | Revised estimation of the cost of the project   |
|    28 | 20.670 | Existing Use                           | Existing use of the building                    |
|    29 | 25.911 | Existing Units                         | Existing number of units                        |
|    30 | 21.336 | Proposed Use                           | Proposed use of the building                    |
|    31 | 25.596 | Proposed Units                         | Proposed number of units                        |
|    32 | 18.757 | Plansets        | Plan representation indicating the general design intent of the foundation..            |
|    33 | 99.998 | TIDF Compliance                        | TIDF compliant or not, this is a new legal requirement           |
|    34 | 21.802 | Existing Construction Type         | Construction type, existing,as categories represented numerically    |
|    35 | 21.802 | Existing Construction Type Description | Descr. of the above, eg.: wood or other construction types       |
|    36 | 21.700 | Proposed Construction Type         | Construction type, proposed, as categories represented numerically   |
|    37 | 21.700 | Proposed Construction Type Description | Description of the above                                         |
|    38 | 97.305 | Site Permit                            | Permit for site                                                  |
|    39 |  0.863 | Supervisor District                    | Supervisor District to which the building location belongs to    |
|    40 |  0.867 | Neighborhoods - Analysis Boundaries    | Neighborhood to which the building location belongs to           |
|    41 |  0.862 | **Zipcode**                            | Zipcode of building address                                      |
|    42 |  0.854 | Location                               | Location in latitude, longitude pair.                            |
|    43 |      0 | Record ID                              | Some ID, not useful for this                                     |

## Understand this code to perform the group imputation:

```python
df = pd.read_csv("titanic/train.csv", index_col='PassengerId')

group_cols = ['Sex','Pclass','Title']
impute_map = df.groupby(group_cols).Age.mean().reset_index(drop=False)

for index, row in impute_map.iterrows(): # Iterate all group possibilities
    
    ind = (df[group_cols] == row[group_cols]).all(axis=1) # Returns Boolean column with the lenth of dataframe        
    
    df[ind] = df[ind].fillna(row["Age"])
```

## Optional External Exercises:

From Kaggle [data cleaning mini course](https://www.kaggle.com/learn/data-cleaning) do:
- [Handling Missing Values](https://www.kaggle.com/alexisbcook/handling-missing-values) Data Cleaning: 1 of 5
- [Inconsistent Data Entry](https://www.kaggle.com/alexisbcook/inconsistent-data-entry) Data Cleaning: 5 of 5
