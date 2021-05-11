<h1 align="center">Predict future sales</h1>


## Introduction

We are going to participate in the [predict-future-sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales) competition from Kaggle. In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms [1C Company](https://1c.ru/eng/title.htm). We are asking you to predict total sales for every product and store in the next month.


## Data


- **`sales_train.csv`** Rows: 2935849 sales (January 2013 -> Octuber 2015)
  - **date**: date in format dd/mm/yyyy.
  - **date_block_num**: a consecutive month number. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
  - **shop_id**: unique identifier of a shop
  - **item_id**: unique identifier of a product
  - **item_price**: current price of an item
  - **item_cnt_day**: number of products sold. You are predicting a monthly amount of this measure.
- **`shops.csv`** Rows: 60 shops
  - **shop_id**
  - **shop_name**: name of shop (RUSSIAN 🇷🇺)
- **`items.csv`** Rows: 22170 products
  - **item_id**
  - **item_name**: name of item (RUSSIAN 🇷🇺)
  - **item_category_id**: unique identifier of item category
- **`item_categories.csv`** Rows: 84 product categories
  - **item_category_id**
  - **item_category_name**: name of item category (RUSSIAN 🇷🇺)
- **`test.csv`** Rows: 214200 pairs combination of (Shop, Item)
  - **ID**: an Id that represents a (Shop, Item) tuple within the test set
  - **shop_id**
  - **item_id**


# Feature Eng

> - Lags
>   - lagA: All previous months
>   - lag1: Previous month
>   - lag2: 2 months ago
>   - lag3: 3 months ago
>   - lag12: 12 months ago (same month of past year)
> - Target: Quantities sold at given month
> - Revenue: price * quantity sold at given month
> - Day of Week Qualities:
>   - %mon: Percentages of sells done on Monday
>   - %tue: Percentages of sells done on Tuesday
>   - %wed: Percentages of sells done on Wednesday
>   - %thu: Percentages of sells done on Thursday
>   - %fri: Percentages of sells done on Friday
>   - %sat: Percentages of sells done on Saturday
>   - %sun: Percentages of sells done on Sunday


### Feature Eng Level 1

Remember to remove duplicated shops and items.


| MONTHS (34)            | SHOPS (60)                | ITEMS                          |
|------------------------|---------------------------|--------------------------------|
| Year (2013..2015)      | Shop id                   | Category id                    |
| Month (1..12)          | Shop city id              | Main Category id               |
| Days of month (28..31) | City latitude             | Subcategory id                 |
| Number of fridays      | City longitude            | SubSubCategory id              |
| Number of saturdays    | Shop type                 | Item id                        |
| Number of sundays      | Physical vs Online        | First  4 chars of item         |
| Number of sat + sun    | Cluster by selling categs | First  6 chars of item         |
| Target_lag1            |                           | First 11 chars of item         |
| Target_lag2            |                           | Mean or median price           |
| Target_lag3            |                           | First day_of_hist sale of item |
| Revenue_lag1           |                           |                                |
| Revenue_lag2           |                           |                                |
| Revenue_lag3           |                           |                                |


### Feature Eng Level 2


| MONTHS x SHOPS         | MONTHS x ITEMS         | SHOPS x ITEMS                          |
|------------------------|------------------------|----------------------------------------|
| %mon_lagA              | Target_lag1            | First day_of_hist sale of item at shop |
| %tue_lagA              | Target_lag2            |                                        |
| %wed_lagA              | Target_lag3            |                                        |
| %thu_lagA              | Revenue_lag1           |                                        |
| %fri_lagA              | Revenue_lag2           |                                        |
| %sat_lagA              | Revenue_lag3           |                                        |
| %sun_lagA              |                        |                                        |



### Feature Eng Level 3


| MONTHS x SHOPS x ITEMS         |
|--------------------------------|
| Target_lagA                    |
| Target_lag1                    |
| Target_lag2                    |
| Target_lag3                    |
| Revenue_lagA                   |
| Revenue_lag1                   |
| Revenue_lag2                   |
| Revenue_lag3                   |





# Construct TRAIN + TEST dataframe

1. Generate useful indexes combination (Month_ID, Shop_ID, Item_id )
   - Solution: Cartesian product of Shop_ID x Item_id for every month
   - Because some shops and some products do not exits at some given moths.
2. LEFT JOIN to add Feature Engeneering columns
3. Add TARGET (Quatities sold of a given item at a given shop at a given month)
4. Get validation set
   - Fold 1:
     - Use months 0...32 for train
     - Use month 33 for validation
   - Fold 2:
     - Use months 0...31 for train
     - Use month 32 for validation
   - Train with all:
     - Use months 0...33 for train
     - Use month 34 for submmit your predictions



| Month_ID | Shop_ID | Item_id   | MONTHS vars | SHOPS vars | ITEMS vars | MONTHSxSHOPS vars | MONTHSxITEMS vars | SHOPSxITEMS vars | MONTHSxSHOPSxITEMS vars | TARGET |
|:--------:|:-------:|:---------:|-------------|------------|------------|-------------------|-------------------|------------------|-------------------------|--------|
| **0**    | **0**   | **0**     |             |            |            |                   |                   |                  |                         |        |
| .        | .       | .         |             |            |            |                   |                   |                  |                         |        |
| .        | .       | .         |             |            |            |                   |                   |                  |                         |        |
| .        | .       | .         |             |            |            |                   |                   |                  |                         |        |
| **34**   | **60**  | **20000** |             |            |            |                   |                   |                  |                         |        |


