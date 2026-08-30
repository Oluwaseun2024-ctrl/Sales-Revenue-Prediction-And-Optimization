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
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Discount%20Distribution.png)

```python
#Create missing indicator
df['discount_missing'] = df['discount'].isnull().astype(int)

#Fill with median
median_discount = df['discount'].median()
df['discount'].fillna(median_discount, inplace=True)
```

**7. Handling Missing Values – Marketing Spend**

Missing values in marketing_spend were addressed using a group-based median imputation strategy to preserve underlying business patterns.

A missing indicator variable (marketing_missing) was created to capture any potential signal associated with absent marketing data.

Instead of applying a global median, missing values were imputed within each product_category group, ensuring that category-specific spending behaviors were maintained.

This approach is more robust and business-aware, as marketing investments typically vary across product categories (e.g., electronics vs. groceries).

Overall, this method improves data quality while maintaining contextual relevance and model interpretability.

```python
#Missing Indicator
df['marketing_missing'] = df['marketing_spend'].isnull().astype(int)

df[['marketing_spend']].hist()
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Marketing%20Spend%20Distribution.png)

```python
#Impute Values
df['marketing_spend'] = df.groupby('product_category')['marketing_spend']\
                         .transform(lambda x: x.fillna(x.median()))
```

**8.Target Variable Validation – Sales Revenue**

The sales_revenue column was validated by confirming its calculation as:

```
df['sales_revenue'] = df['units_sold'] * df['price_per_unit']
```

This ensures that the target variable accurately reflects total revenue generated per transaction. Verifying this relationship is critical to maintain data integrity and ensure reliable model training.

## FEATURE ENGINEERING

**1. Time-Based Features**

Time-based features were created to capture temporal patterns in sales behavior.

The dataset was first sorted by date to ensure chronological consistency, which is critical for time-dependent features and any future time series modeling.

The following features were extracted from the date column:

- year: Captures long-term trends over time
- day_of_week: Identifies weekday patterns in sales
- quarter: Helps model seasonal business cycles
- is_weekend: Binary indicator to distinguish weekend vs. weekday behavior

These features enable the model to learn seasonality, weekly trends, and temporal shifts in customer purchasing patterns.

```Python
# Ensure date is sorted (VERY IMPORTANT for time features)
df = df.sort_values(by='date')

# Extract time features
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
```

**2. Lag Features (Core of Forecasting)**

Lag features were engineered to incorporate historical sales information, which is essential for time-aware predictive modeling.

The dataset was sorted by store_id, product_category, and date to maintain proper temporal order within each group.

Lag features were created per group (store + product category) to avoid data leakage and ensure accurate temporal relationships:

- lag_1: Sales revenue from the previous day
- lag_7: Sales revenue from the same day in the previous week
- lag_14: Sales revenue from two weeks prior

This approach reflects real-world sales dynamics:

- Recent sales trends influence current performance
- Weekly patterns capture recurring customer behavior

These features form the core foundation for forecasting, enabling the model to learn from historical dependencies rather than relying solely on static inputs.

```python
df = df.sort_values(by=['store_id', 'product_category', 'date'])

df['lag_1'] = df.groupby(['store_id', 'product_category'])['sales_revenue'].shift(1)
df['lag_7'] = df.groupby(['store_id', 'product_category'])['sales_revenue'].shift(7)
df['lag_14'] = df.groupby(['store_id', 'product_category'])['sales_revenue'].shift(14)
```

**3. Rolling Features (Trend Awareness)**

Rolling features were engineered to capture short-term trends and variability in sales over time.

The following features were created using a 7-day rolling window within each store_id and product_category group:

- rolling_mean_7: Average sales revenue over the past 7 days
- rolling_std_7: Standard deviation of sales revenue over the past 7 days

A shift(1) was applied before the rolling calculation to ensure that only past data is used, preventing data leakage and maintaining proper forecasting integrity.

These features provide:
- Trend signals (via rolling mean)
- Volatility insights (via rolling standard deviation)

After generating lag and rolling features, rows with resulting missing values (due to shifting and windowing) were removed to ensure a clean dataset for modeling.

This step enhances the model’s ability to understand recent performance trends and fluctuations, which are critical for accurate time-series forecasting.

**4. Business-Driven Features**

Domain-specific features were engineered to incorporate business logic and improve the model’s ability to capture real-world revenue dynamics.

**Revenue Drivers**

Both unit-level and transaction-level pricing features were created to reflect pricing strategy and its impact on revenue:
- price_after_discount_unit: Effective price per unit after applying discount
- net_revenue: Adjusted revenue after discount

```Pyhton
df['price_after_discount_unit'] = df['price_per_unit'] * (1 - df['discount'])
df['net_revenue'] = df['sales_revenue'] * (1 - df['discount'])
```
These features allow the model to learn price sensitivity and discount-driven purchasing behavior.

**Discount Intensity**

discount_intensity was introduced as:
```python
Discount × Units Sold
```
This captures the magnitude of discount impact, combining both pricing strategy and sales volume.
```python
df['discount_intensity'] = df['discount'] * df['units_sold']
```

**Marketing Efficiency**

marketing_efficiency was defined as: Revenue generated per unit of marketing spend

A small constant (+1) was added to the denominator to prevent division-by-zero errors.
```python
df['marketing_efficiency'] = df['sales_revenue'] / (df['marketing_spend'] + 1)
```
This feature helps quantify return on marketing investment (ROMI).

**Demand Intensity**

avg_unit_price was calculated as: Revenue per unit sold

A small constant (+1) ensures numerical stability.
```python
df['avg_unit_price'] = df['sales_revenue'] / (df['units_sold'] + 1)
```
This provides insight into effective selling price and demand behavior under varying conditions.

Overall, these features embed business intuition into the dataset, enabling the model to move beyond raw data and learn economic relationships driving sales performance.

## EXPLORATORY DATA ANALYSIS (EDA)

**STEP 1: KPIs (Key Performance Indicator**

**Total Revenue**
```python
total_sales_revenue = df['sales_revenue'].sum()
print(f"Total Sales Revenue: {total_sales_revenue:.2f}")
Total Sales Revenue: 2362239427.33
```

**Total Net Revenue**
```python
total_net_revenue = df['net_revenue'].sum()
print(f"Total Net Revenue: {total_net_revenue:.2f}")
Total Net Revenue: 1987564418.28
```

**Revenue Reduction due to Discounts**
```python
revenue_reduction = total_sales_revenue - total_net_revenue
print(f"Revenue Reduction due to Discounts: {revenue_reduction:.2f}")
Revenue Reduction due to Discounts: 374675009.05
```

**Total Unit Sold**
```python
total_units_sold = df['units_sold'].sum()
print(f"Total Units Sold: {total_units_sold}")
Total Units Sold: 9299121
```

**STEP 2: Target Variable Analysis (Sales Revenue)**

**Distribution**
```python
import matplotlib.pyplot as plt
df['sales_revenue'].hist(bins=50)
plt.title('Sales Revenue Distribution')
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Sales%20Revenue%20Distribution.png)

The sales_revenue variable exhibits a right-skewed distribution, with the majority of observations concentrated at lower values and a long tail extending toward higher revenue values.

**Summary Statistics**
```Python
df['sales_revenue'].describe()
```
```
Count: 35,800
Mean: 65,984
Median (50%): 52,073
Standard Deviation: 55,949
Minimum: 64.91
25th Percentile: 25,015
75th Percentile: 90,505
Maximum: 510,570
```
**Observation:** The mean exceeds the median, indicating positive skewness. The wide range and high standard deviation suggest substantial variability in sales revenue across observations.

**STEP 3: Time Series Analysis**

**Daily Trend**
```python
df.groupby('day_of_week')['sales_revenue'].sum().plot(figsize=(12,5))
plt.title('Daily Sales Trend')
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Daily%20Sales%20Trend.png)

**Observation:** Sales revenue remains relatively stable during weekdays (Monday–Friday), with a noticeable increase on Saturday and Sunday, indicating higher revenue concentration over the weekend.

**Average Monthly Sales**
```python
df['month'] = df['date'].dt.month
df.groupby('month')['sales_revenue'].mean().plot(kind='line')
plt.title('Average Monthly Sales')
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Average%20Monthly%20Sales.png)

**Observation:** Sales revenue shows a gradual upward trend throughout the year, with a significant spike in December, indicating strong seasonality and year-end sales concentration.

**STEP 4: Category & Store Performance**

**Product Category**
```python
df.groupby('product_category')['sales_revenue'].sum().sort_values().plot(kind='bar')
plt.title('Revenue by Product Category')
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Revenue%20By%20Product%20Category.png)

**Observation:** Electronics contributes the highest share of total revenue, followed by Home and Clothing. Groceries generate the lowest revenue among all categories.

**Store Location**
```python
df.groupby('store_location')['sales_revenue'].mean().plot(kind='bar')
plt.title('Average Revenue by Store')
plt.show()
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Average%20Revenue%20By%20Store%20Location.png)

**Observation:** Average sales revenue is relatively consistent across all locations, with Lagos and Abuja showing marginally higher values. Port Harcourt records the lowest average, though the variation across locations is minimal.

**STEP 5: Pricing & Discount Impact**

**Discount vs Revenue**
```python
df[['discount', 'sales_revenue']].corr()
```
|              | discount | sales_revenue |
|--------------|----------|---------------|
| discount     | 1.000000 | 0.121043      |
| sales_revenue| 0.121043 | 1.000000      |

**Observation:** The correlation between discount and sales revenue is positive but weak (0.12), indicating that higher discounts are associated with a slight increase in revenue, though the relationship is not strong.

**Price vs Units Sold**
```python
df[['price_per_unit', 'units_sold']].corr()
```
|              | price_per_unit | units_sold |
|--------------|----------------|------------|
| price_per_unit | 1.000000       | 0.004562   |
| units_sold     | 0.004562       | 1.000000   |

**Observation:** There is no meaningful linear relationship between price per unit and units sold, as indicated by the near-zero correlation. This suggests that changes in price do not significantly impact sales volume in a linear manner.

**Note:** The near-zero correlation between price and units sold may seem counterintuitive, but it does not indicate an error in the analysis. Instead, it suggests that within this dataset, price alone does not have a strong linear influence on sales volume—likely due to factors such as limited price variation, aggregation across different products, and the influence of other drivers like seasonality and demand patterns.

**STEP 6: Marketing Impact**

**Marketing Spend vs Revenue**
```python
df[['marketing_spend', 'sales_revenue']].corr()
```
|                 | marketing_spend | sales_revenue |
|-----------------|-----------------|---------------|
| marketing_spend | 1.000000        | 0.102915      |
| sales_revenue   | 0.102915        | 1.000000      |

**Observation:** There is a weak positive relationship between marketing spend and sales revenue. This suggests that higher marketing investment is associated with a slight increase in revenue, but the effect is not strong.

**Note:** The weak correlation does not imply that marketing is ineffective; rather, it indicates that marketing spend alone does not strongly explain revenue changes in this aggregated dataset. Other factors such as timing, campaign quality, product demand, and seasonality may play a more significant role, and the impact of marketing may not be strictly linear or immediate.

## ENCODING CATEGORICAL COLUMNS

To prepare the dataset for modeling, categorical features were converted into numerical format using One-Hot Encoding.
```python
#One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=['product_category', 'store_location', 'season'],
    drop_first=True
)
```
Categorical variables were encoded using one-hot encoding to transform them into a machine-readable format. One category from each feature was dropped to serve as a baseline, preventing redundancy and ensuring stable model estimation.

**Encoding Store ID**
```python
df = pd.get_dummies(df, columns=['store_id'], drop_first=True)
```
The store_id variable was treated as a categorical feature and encoded using one-hot encoding to capture store-level differences in performance. One category was dropped to act as a baseline, ensuring efficient and stable model estimation.

## FEATURE SELECTION PREP

**Target Variable Definition**

The target variable for the modeling process was defined as:
```python
y = df['sales_revenue']
```
This variable represents the total revenue generated and serves as the dependent variable that the machine learning models aim to predict.

**Removing Data Leakage (Critical Step)**
```python
# REMOVED THESE:
cols_to_drop = [
    'sales_revenue',        # target
    'net_revenue',          # derived from target
    'marketing_efficiency', # uses target → leakage
    'avg_unit_price'        # uses target → leakage
]
x = df.drop(columns=cols_to_drop)
```
Data leakage occurs when input features contain information that would not be available at prediction time or are directly derived from the target. Including such variables leads to overly optimistic model performance and poor real-world generalization.

Columns that directly or indirectly incorporate the target variable were removed to prevent data leakage. This ensures that model performance remains realistic and generalizable, reflecting true predictive capability rather than artificially inflated accuracy.

**Removing Non-Useful Columns**
```python
# Drop raw date column
x = x.drop(columns=['date'])
```
The raw date column was removed as it does not provide usable information in its original format. Without proper feature engineering (e.g., extracting temporal components), it would not contribute meaningfully to model performance.

## TRAIN TEST SPLIT
```python
# Step 1: Ensure data is sorted
df = df.sort_values('date')
```
```python
# Step 2: Align x and y
x = x.loc[df.index]
y = y.loc[df.index]
```
```python
# Step 3: Time-based split (80% train, 20% test)
split_index = int(len(x) * 0.8)

x_train = x.iloc[:split_index]
x_test  = x.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]
```

This dataset contains temporal structure (date, lag features, rolling metrics), making it a time series prediction problem.

A random split would shuffle observations, allowing the model to learn from future data → data leakage.

The time-based split ensures:

- Training uses only historical data
- Testing simulates future, unseen data

A time-based split was used instead of random sampling to preserve the chronological order of observations. This approach prevents data leakage and ensures that model evaluation reflects real-world forecasting conditions, where predictions are made on future data not available during training.

## SCALING
```python
# Creating scaled copies
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()
```
```python
# Define numerical columns to scale
num_cols = [
    'units_sold',
    'price_per_unit',
    'discount',
    'marketing_spend',
    'economic_index',
    'lag_1',
    'lag_7',
    'lag_14',
    'rolling_mean_7',
    'rolling_std_7',
    'price_after_discount_unit',
    'discount_intensity'
]
```
```python
# Apply scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test_scaled[num_cols] = scaler.transform(x_test[num_cols])
```

Standardization rescales numerical features to have:
- Mean = 0
- Standard deviation = 1

Scaling is applied only to continuous numerical variables because:
- These features vary in magnitude and units
- Many models (e.g., linear regression, KNN, SVM) are scale-sensitive

The following are excluded from scaling:
- One-hot encoded variables (0/1)
- Binary indicators
- Encoded categorical features

Scaling them would:
- Distort their meaning
- Reduce interpretability
- Provide no modeling benefit

The scaler is:
- Fit on training data only → prevents leakage
- Applied to test data using the same transformation

Scaling was selectively applied to continuous numerical features to standardize their ranges while preserving the interpretability of binary and categorical variables. The scaler was fit exclusively on the training data and then applied to the test set to prevent data leakage and ensure consistent transformation.
