# Sales-Revenue-Prediction-And-Optimization
A machine learning–based system that predicts sales revenue and identifies key business drivers to support data-driven decision-making, deployed through an interactive Streamlit application.

## TABLE OF CONTENT
1.	Project Overview
2.	Business Problem
3.	Project Objectives
4.	Domain Knowledge
5.	Dataset Description
6.	Data Dictionary
7.	Data Cleaning
8.	Feature Engineering
9.	Exploratory Data Analysis (EDA)
10.	Encoding Categorical Columns
11.	Feature Selection Prep
12.	Train Test Split
13.	Scaling
14.	Modelling Approach And Evaluation 
15.	Results & Insights
16.	Business Recommendations
17.	Model Deployment
18.	How To Run the App
19.	Conclusion

## PROJECT OVERVIEW
This project aims to predict sales revenue using historical retail data by applying machine learning techniques. It involves data cleaning, feature engineering, and model development to identify key drivers of sales.

A baseline Linear Regression model and advanced models (Random Forest and XGBoost) were trained and evaluated. The optimized XGBoost model was selected as the final model due to its superior performance.

The outcome provides both accurate predictions and actionable insights into factors such as pricing, discounts, marketing, and seasonality.

## BUSINESS PROBLEM
Retail businesses often struggle to accurately forecast sales due to the dynamic influence of factors such as pricing, discounts, marketing spend, and seasonality. Poor forecasting can lead to overstocking, stockouts, and inefficient allocation of marketing resources.

The core problem is to predict sales revenue accurately and identify the key drivers influencing it, enabling businesses to optimize pricing strategies, plan promotions effectively, and improve overall revenue performance.

## PROJECT OBJECTIVES
- Develop a machine learning model to accurately predict sales revenue
- Identify and quantify key factors influencing sales performance
- Engineer relevant features to capture pricing, promotional, and temporal effects
- Compare baseline and advanced models to select the best-performing approach
- Generate actionable insights to support pricing, marketing, and inventory decisions

## DOMAIN KNOWLEDGE
Sales revenue in retail is influenced by a combination of pricing strategy, promotional activities, customer demand patterns, and seasonality. Discounts and marketing campaigns typically drive short-term sales increases but may impact profit margins if not optimized.

Demand often varies across time due to factors such as holidays, weekends, and seasonal trends, making time-based features critical for accurate forecasting. Additionally, external and operational factors such as product availability, store performance, and competitive pricing can significantly affect revenue outcomes.

Understanding these dynamics is essential for building effective predictive models and generating insights that align with real-world business behaviour.

## DATASET DESCRIPTION
The dataset is a synthetic retail dataset generated specifically for the purpose of building and evaluating a sales revenue prediction model. It simulates real-world retail operations and customer behavior.

It contains a mix of numerical, categorical, and time-based features, including:
- Sales & Pricing: units sold, price per unit, discount
- Marketing: marketing spends
- Time Features: day of week, month, year, quarter, season, weekend indicator
- Economic Indicator: economic index
- Engineered Features: lag variables (lag_1, lag_7, lag_14), rolling statistics, discount intensity, price after discount#
- Categorical Encodings: product category, store location, and store ID (one-hot encoded)
Missing values in key variables such as discount and marketing spend were handled using indicator flags and appropriate imputation.

Overall, the dataset is designed to capture realistic sales patterns, temporal dependencies, and business drivers, enabling effective model development and analysis.

## DATA DICTIONARY
The dataset contains 36,500 rows and 14 original columns. Below is a description of each variable:

| Column Name       | Data Type        | Description                                                                 |
|-------------------|------------------|-----------------------------------------------------------------------------|
| date              | datetime         | The transaction date for each record                                        |
| store_id          | integer          | Unique identifier for each store                                            |
| product_category  | categorical      | Category of product (Electronics, Clothing, Groceries, etc.)               |
| units_sold        | float            | Number of units sold (influenced by demand)                         |
| price_per_unit    | float            | Selling price per unit of product                                           |
| discount          | float            | Discount applied (0 to 0.3 range; may contain missing values)              |
| marketing_spend   | float            | Daily marketing expenditure allocated to the product/store                  |
| holiday_flag      | binary (0/1)     | Indicates whether the day is a holiday period                               |
| day_of_week       | integer          | Day of the week (0 = Monday, 6 = Sunday)                                    |
| month             | integer          | Month of the year (1–12)                                                    |
| season            | categorical      | Derived seasonal grouping (Harmattan, Rainy, etc.)                          |
| store_location    | categorical      | City where the store is located                                             |
| economic_index    | float            | Simulated economic indicator reflecting market conditions                  |
| sales_revenue     | float            | Target variable — total revenue generated                                  |

This data dictionary provides a clear understanding of the dataset structure and supports effective preprocessing, feature engineering, and modelling.

## DATA CLEANING
**1. Converted the Date Column**

The date column was converted to a proper datetime format to enable time-based analysis and feature extraction:
```Python
df['date'] = pd.to_datetime(df['date'])
```
This step ensures that the dataset supports accurate temporal operations such as extracting day, month, year, and generating time-based features for modelling.

**2. Converted Categorical Columns**

Categorical variables were explicitly converted to the appropriate data type to ensure efficient memory usage and proper handling during analysis and modelling:
```Python
categorical_cols = ['product_category', 'season', 'store_location']
for col in categorical_cols:
    df[col] = df[col].astype('category')
```
This step ensures that categorical features are correctly recognized, improving downstream processes such as encoding and model training.

**3. Checked for Duplicates**

A check for duplicate records was performed to ensure data integrity:
```Python
df.duplicated().sum()
```
No duplicate rows were found in the dataset, confirming that all records are unique and suitable for analysis.

**4. Statistical Summary**

A summary statistics check was conducted to understand the distribution and quality of the dataset:
```Python
df.describe()
```
**Key Observations:**
- The dataset contains 36,500 records across all key variables

Missing values are present in:
- discount (730 missing)
- marketing_spend (730 missing)
- units_sold and sales_revenue show high variability, indicating potential outliers and demand fluctuations
- price_per_unit and discount fall within expected retail ranges
- holiday_flag is highly imbalanced (mostly non-holiday days)
- economic_index shows relatively stable variation, representing external economic conditions

This step provided an initial understanding of data distribution, missing values, and potential areas requiring further preprocessing.

**5. Missing Values Assessment**

The dataset was evaluated for missing values using 
```Python
df.isnull().sum().
```
The variables discount and marketing_spend each contain 730 missing values.

All other variables have no missing values, indicating overall good data completeness.

This pattern suggests that missing values are isolated to promotional-related features and may indicate periods with no recorded discount or marketing activity rather than random data loss.

**6. Handling Missing Values – Discount Column**

Missing values in discount account for approximately 2% of the dataset (730 rows), which is relatively small but still significant for modeling.

A missing value indicator (discount_missing) was created to capture any potential signal associated with missingness.

The missing values were then imputed using the median, ensuring robustness against skewed distributions and preserving the overall data structure.

This approach aligns with industry best practices by both retaining all observations and preserving potential information in missing patterns.
```Python
df['discount'].isnull().mean()
```
np.float64(0.02)

```Python
df[['discount']].hist()
```
![]()
