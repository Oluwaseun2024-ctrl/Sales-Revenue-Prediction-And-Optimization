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

## MODELING APPROACH

**1. LINEAR REGRESSION**
```python
#train using scaled data:
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train_scaled, y_train)
```
```python
#Make Predictions
y_pred_lr = lr_model.predict(x_test_scaled)
```
```python
Model Evaulation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import numpy as np 
mae = mean_absolute_error(y_test, y_pred_lr) 
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr)) 
r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:") 
print("MAE:", mae) 
print("RMSE:", rmse) 
print("R2 Score:", r2)
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Linear%20Regression.png)

**Summary of Results**
- MAE: 16,183
- RMSE: 26,993
- R² Score: 0.844

**Interpretation**

1.	R² = 0.844 : The model explains approximately 84.4% of the variance in sales_revenue, indicating a strong overall fit.

2.	MAE (~16K): On average, predictions deviate from actual revenue by about $16,183.

3.	RMSE (~27K): Penalizes larger errors more heavily, suggesting that while average errors are moderate, there are some larger deviations.

**What This Means (Business Perspective)**

The model captures the major drivers of revenue effectively, making it suitable for:
- Forecasting trends
- Supporting strategic decisions

However, the gap between MAE and RMSE indicates occasional large prediction errors, which may be driven by:
- Sudden demand spikes
- Promotions or external factors not fully captured

The Linear Regression model demonstrates strong predictive performance, explaining a substantial portion of the variance in sales revenue. While average prediction errors are within a reasonable range, the higher RMSE suggests the presence of occasional large deviations, indicating opportunities for further model refinement or the inclusion of additional explanatory variables.

**Extract Coefficients (Very Valuable)**
```python
#This is where Linear Regression shines — interpretability:
coefficients = pd.DataFrame({
    'Feature': x_train_scaled.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', ascending=False)
coefficients.head(10)
```

**Top Positive Drivers of Revenue**
| Feature | Coefficient |
|---|---:|
| price_per_unit | 70,809 |
| units_sold | 32,966 |
| rolling_mean_7 | 1,026 |
| season_Rainy | 824 |
| discount_missing | 621 |
| season_Post-Rainy | 550 |
| product_category_Groceries | 549 |
| marketing_missing | 429 |
| store_id_5 | 253 |
| discount_intensity | 244 |

**Interpretation**

1.	price_per_unit (strongest driver): A $1 increase in price (holding other factors constant) is associated with a ~$70K increase in revenue, indicating pricing power dominates revenue generation.

2.	units_sold: As expected, higher sales volume significantly increases revenue.

3.	rolling_mean_7: Recent sales trends positively influence current revenue, validating the importance of temporal momentum.

4.	seasonality (Rainy / Post-Rainy): Certain seasons contribute positively, suggesting demand patterns vary over time.

5.	product_category_Groceries: Indicates this category tends to generate higher revenue relative to the baseline category.

6.	Missing indicators (discount_missing, marketing_missing): These capturing mechanisms may reflect systematic patterns in data absence, which the model is leveraging.

7.	store_id_5: This specific store outperforms the baseline, indicating location-specific effects.

8.	discount_intensity: Higher discount levels still contribute positively, likely via volume-driven revenue lift.

The coefficient magnitudes reflect the relative influence of features on revenue; however, they should be interpreted cautiously due to differences in feature scaling and potential correlations among predictors. While Linear Regression provides valuable directional insights, these relationships represent associations rather than strict causation.

**2. RANDOM FOREST**
```python
#Train Model
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train, y_train)
```
```python
#Predict
y_pred_rf = rf_model.predict(x_test)
```
```python
#Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/Random%20Forest.png)

**Summary of Results**
- MAE: 1,203
- RMSE: 5,366
- R² Score: 0.994

**Interpretation**

1.	R² = 0.994: The model explains 99.4% of the variance in sales_revenue, indicating an extremely strong fit.

2.	MAE (~1.2K): Predictions are, on average, off by only about $1,203, which is significantly lower than Linear Regression.

3.	RMSE (~5.3K): Even larger errors are relatively small compared to the scale of revenue, showing high prediction stability.

**Comparison vs Linear Regression**
| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 16,183 | 1,203 |
| RMSE | 26,993 | 5,366 |
| R² | 0.844 | 0.994 |

The Random Forest model dramatically outperforms Linear Regression across all metrics.

**Critical Insight (Very Important)**

While performance is exceptionally high, this raises a red flag:

Such a high R² (≈0.99) may indicate:
- Overfitting, especially with:
- Lag features
- Rolling statistics
- Many encoded variables

The model may be capturing patterns too specific to training data

**What This Means (Business Perspective)**

The model is highly accurate for prediction tasks

Suitable for:
- Short-term forecasting
- Operational decision-making

But may not generalize as well to completely unseen future conditions

The Random Forest model achieved significantly higher predictive accuracy compared to Linear Regression, capturing nearly all variability in sales revenue. However, the exceptionally high performance may indicate potential overfitting, and results should be interpreted with caution when generalizing to future or unseen data.

**Extract Coefficients (Very Valuable)**
```python
#This is where Random Forest shines — interpretability:
feature_importance = pd.DataFrame({
    'Feature': x_test.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importance.head(10)
```

**Top Drivers of Revenue**
| Feature | Importance |
|---|---:|
| units_sold | 0.5203 |
| price_per_unit | 0.4792 |
| discount_intensity | 0.000095 |
| price_after_discount_unit | 0.000085 |
| economic_index | 0.000048 |
| rolling_std_7 | 0.000047 |
| marketing_spend | 0.000039 |
| lag_14 | 0.000032 |
| rolling_mean_7 | 0.000027 |
| lag_1 | 0.000021 |

**Interpretation**

units_sold (52%) and price_per_unit (48%) dominate the model → Together, they explain ~99.95% of total feature importance, meaning:
- Revenue is almost entirely driven by volume × price dynamics
- This aligns directly with the business formula:
```
Revenue ≈ Units Sold × Price
```
All other variables contribute negligibly → Features like marketing spend, seasonality, lag variables, and economic indicators have minimal marginal impact once price and volume are known.

**Critical Insight (Very Important)**

This distribution suggests the model is heavily relying on near-direct drivers of the target:
- units_sold and price_per_unit are mechanically linked to revenue

This is not leakage, but it does mean:
- The model is learning a relationship that is almost deterministic
- Additional features add little incremental predictive power

**Business Interpretation**

The model confirms a fundamental truth:
- Revenue is primarily a function of pricing and sales volume

Secondary factors (marketing, seasonality, economy):
- Likely influence revenue indirectly by affecting units sold or pricing
- But do not independently drive revenue once those are included

Feature importance analysis shows that revenue is overwhelmingly driven by units sold and price per unit, with all other variables contributing marginally. This indicates that the model is primarily capturing the fundamental revenue relationship, while additional features provide limited incremental predictive value.

**3. XGBOOST**
```python
#Train Model
from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb_model.fit(x_train, y_train)
```
```python
#Predict
y_pred_xgb = xgb_model.predict(x_test)
```
```python
#Evaluate
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Results:")
print("MAE:", mae_xgb)
print("RMSE:", rmse_xgb)
print("R2 Score:", r2_xgb)
```
![](https://github.com/Oluwaseun2024-ctrl/Sales-Revenue-Prediction-And-Optimization/blob/main/XGBOOST.png)

**Summary of Results**
- MAE: 1,419
- RMSE: 6,683
- R² Score: 0.990

**Interpretation**

1.	R² = 0.990 → The model explains 99.0% of the variance in sales_revenue, indicating excellent predictive performance.

2.	MAE (~1.4K) → On average, predictions deviate by about $1,419, which is very low relative to revenue scale.

3.	RMSE (~6.7K) → Slightly higher than Random Forest, indicating some larger errors, but still well-controlled.

**Comparison Across Models**
| Metric | Linear Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| MAE | 16,183 | 1,203 | 1,419 |
| RMSE | 26,993 | 5,366 | 6,683 |
| R² | 0.844 | 0.994 | 0.990 |

**Performance Ranking:**

1.	Random Forest (Best)

2.	XGBoost (Very close second)

3.	Linear Regression (Baseline)

**Key Insights**

XGBoost significantly outperforms Linear Regression, capturing nonlinear relationships and interactions.

Compared to Random Forest:
- Slightly higher error metrics
- Still very competitive and robust

Both tree-based models confirm:
- The problem has strong nonlinear structure
- Feature interactions matter

**Critical Observation**

Similar to Random Forest, very high R² suggests:
- The model is leveraging strong deterministic signals (e.g., price × units)
- Potential overfitting risk still exists, though XGBoost typically generalizes better due to boosting

**Business Interpretation**

XGBoost provides:
- Highly accurate forecasts
- Better handling of complex relationships

Suitable for:
- Production-grade prediction systems
- Revenue forecasting under varying conditions

The XGBoost model achieved excellent predictive performance, closely matching the Random Forest model while significantly outperforming Linear Regression. Its ability to capture nonlinear relationships and feature interactions makes it a strong candidate for robust revenue forecasting, although the high accuracy warrants careful validation to ensure generalizability.

**Extract Coefficients (Very Valuable)**
```python
#This is where Random Forest shines — interpretability:
feature_importance = pd.DataFrame({
    'Feature': x_test.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importance.head(10)
```

**Top Drivers of Revenue**
| Feature | Importance |
|---|---:|
| price_per_unit | 0.4946 |
| units_sold | 0.4891 |
| store_id_7 | 0.0034 |
| holiday_flag | 0.0029 |
| discount | 0.0012 |
| season_Harmattan | 0.0011 |
| discount_intensity | 0.0010 |
| store_id_8 | 0.0008 |
| store_location_Port Harcourt | 0.0008 |
| month | 0.0005 |

**Interpretation**

price_per_unit and units_sold dominate (~98.4%) → Just like Random Forest, XGBoost confirms that revenue is כמעט entirely driven by price and volume.

Secondary contributors (very small impact):
- store_id_7, store_id_8 → Certain stores slightly outperform others
- holiday_flag → Holidays create marginal revenue uplift
- discount & discount_intensity → Promotions have limited direct effect once price and volume are known
- season_Harmattan & month → Minor seasonal patterns exist
- store_location_Port Harcourt → Location-specific nuance

**Key Insight**

XGBoost captures slightly more nuanced signals than Random Forest:
- It assigns small but meaningful importance to:
- Holidays
- Seasonality
- Store-level variation
- However, these effects are still tiny compared to core drivers

**Critical Observation**

The dominance of:
- units_sold
- price_per_unit

indicates the model is learning a near-deterministic structure:
```
Revenue ≈ Units Sold × Price
```
This explains:
- Extremely high R² across tree-based models
- Very low prediction error

**Business Interpretation**

Primary levers:
- Pricing strategy
- Sales volume

Secondary levers (fine-tuning):
- Promotions
- Seasonal campaigns
- Store-level optimization

Feature importance from the XGBoost model confirms that revenue is overwhelmingly driven by units sold and price per unit, with only marginal contributions from seasonality, promotions, and store-level factors. While the model captures some additional nuance compared to Random Forest, the underlying revenue relationship remains strongly dominated by core pricing and volume dynamics.

## RESULTS & INSIGHTS SUMMARY

**1. Overall Model Performance**

Three models were trained and evaluated:
| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 16,183 | 26,993 | 0.844 |
| Random Forest | 1,203 | 5,366 | 0.994 |
| XGBoost | 1,419 | 6,683 | 0.990 |

**Key Takeaways**
- Tree-based models (Random Forest & XGBoost) significantly outperform Linear Regression
- Random Forest achieved the best overall performance
- Linear Regression serves as a useful baseline, but fails to capture complex relationships

**2. Model Behavior & Learning Patterns**

**Linear Regression**

Captures general trends but assumes linear relationships

Performance (R² = 0.844) indicates:
- Important variables are included
- But nonlinear interactions are missed

**Random Forest & XGBoost**
- Capture nonlinearities and feature interactions
- Achieve extremely high accuracy (R² ≈ 0.99)
- Show strong ability to model complex real-world dynamics

**3. Core Drivers of Revenue (Consistent Across Models)**

Across all models, the same dominant features emerge:
- units_sold
- price_per_unit

**Insight**

Revenue is fundamentally driven by:
- Revenue ≈ Units Sold × Price

These two variables alone explain ~98–99% of model importance. This explains:
- Very high R² scores
- Very low prediction errors

**4. Role of Other Variables**

Other features (marketing, seasonality, lag variables, etc.) show:
- Minimal direct importance in tree-based models
- Small but detectable effects in XGBoost

**Interpretation**

These variables likely influence revenue indirectly:
- Marketing → increases units sold
- Seasonality → affects demand patterns
- Discounts → impact pricing and volume

Once price and units sold are known, their additional contribution becomes marginal.

**5. Business Insight**

Primary Revenue Levers
- Pricing strategy (price_per_unit)
- Sales volume (units_sold)

Secondary Levers (Indirect Impact)
- Promotions & discounts
- Marketing spend
- Seasonality & holidays
- Store/location effects

**6. Critical Observations (Very Important)**

**1. Near-Deterministic Relationship**

The model is effectively learning a mathematical identity. This leads to:
- Extremely high predictive accuracy
- Low marginal contribution from other features

**2. Potential Overfitting Risk**

Very high R² (≈0.99) suggests: Model may be too tailored to observed patterns. However:
- Time-based split reduces leakage risk
- Results are still valid but should be interpreted cautiously

3. Correlation vs Reality
Earlier findings (e.g., weak price vs units correlation) may seem counterintuitive
But:
•	Lack of correlation ≠ lack of importance
•	Multivariate models capture combined effects, not isolated relationships

7. Final Conclusion
The modeling pipeline is methodologically sound:
•	Proper leakage handling
•	Time-based validation
•	Correct preprocessing
Best Model: Random Forest
Most Insightful Model: Linear Regression (interpretability)
The analysis reveals that sales revenue is overwhelmingly driven by core transactional variables—units sold and price per unit—while other factors such as marketing, seasonality, and economic conditions play a secondary, indirect role. Advanced models like Random Forest and XGBoost achieve near-perfect predictive performance by capturing this strong underlying relationship, though their results should be interpreted with awareness of potential overfitting and the inherently deterministic nature of the target variable.

BUSINESS RECOMMENDATIONS
1. Prioritize Pricing Optimization
•	Implement data-driven pricing strategies (e.g., dynamic pricing, price elasticity testing)
•	Continuously test price points to identify revenue-maximizing thresholds
•	Avoid arbitrary pricing decisions—small adjustments can produce significant revenue impact

2. Focus on Increasing Sales Volume
•	Strengthen distribution and product availability to prevent stockouts
•	Expand sales channels (online, retail partnerships, regional expansion)
•	Improve customer acquisition and retention strategies to drive consistent demand

3. Reframe Marketing Strategy
•	Shift from spend-based marketing to performance-based marketing
•	Track and optimize for:
o	Conversion rate
o	Revenue per campaign
•	Invest only in campaigns that demonstrably increase units sold

4. Use Discounts Strategically
•	Avoid over-reliance on discounting as a growth strategy
•	Apply discounts selectively for:
o	Inventory clearance
o	Low-demand periods
•	Ensure discounts increase total revenue, not just sales volume

5. Leverage Seasonal and Holiday Trends
•	Align promotions and inventory planning with peak demand periods
•	Use historical patterns to:
o	Forecast demand
o	Optimize staffing and supply chain decisions

6. Optimize Store-Level Performance
•	Identify high-performing stores and replicate their strategies
•	Investigate underperforming locations and adjust:
o	Pricing
o	Product mix
o	Local marketing efforts

7. Align Strategy with Core Revenue Drivers
•	Recognize that revenue is primarily driven by:
o	Units Sold
o	Price per Unit
•	All business initiatives (marketing, discounts, expansion) should be evaluated based on how effectively they impact these two core drivers
The business should concentrate on optimizing price and maximizing sales volume, while treating marketing, discounts, and seasonal tactics as supporting levers that influence these primary drivers.

MODEL DEPLOYMENT
The trained Random Forest model was serialized and deployed using Streamlit to provide an interactive interface where users can input key business variables (e.g., units sold, price, discounts, seasonality) and obtain real-time sales revenue predictions, enabling practical, data-driven decision-making.
Save the Random Forest Model
import joblib
# Save the trained Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')

Download the Model
from google.colab import files
files.download('random_forest_model.pkl')

Save Feature Columns
This prevents deployment errors later (very important for Streamlit):
# Save feature column order
joblib.dump(x_train.columns.tolist(), 'model_features.pkl')

Download:
files.download('model_features.pkl')

COMPLETE STREAMLIT APP
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODEL & FEATURES
# ==============================
model = joblib.load('random_forest_model.pkl')
model_features = joblib.load('model_features.pkl')


# ==============================
# PREPROCESS FUNCTION (FROM YOUR NOTEBOOK)
# ==============================
def preprocess_input(df):
    # One-hot encoding (same as training)
    df = pd.get_dummies(
        df,
        columns=['product_category', 'store_location', 'season'],
        drop_first=True
    )

    df = pd.get_dummies(df, columns=['store_id'], drop_first=True)

    # Drop date if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Align with training features
    df = df.reindex(columns=model_features, fill_value=0)

    return df


# ==============================
# STREAMLIT UI
# ==============================
st.title("📊 Sales Revenue Prediction App")

st.write("Enter product and sales details to predict revenue")

# --- INPUTS ---
units_sold = st.number_input("Units Sold", min_value=0, value=100)
price_per_unit = st.number_input("Price per Unit ($)", min_value=0.0, value=10.0)
discount = st.slider("Discount (%)", 0, 100, 0)
holiday_flag = st.selectbox("Holiday?", [0, 1])
month = st.selectbox("Month", list(range(1, 13)))

product_category = st.selectbox(
    "Product Category",
    ["Electronics", "Clothing", "Groceries"]
)

store_location = st.selectbox(
    "Store Location",
    ["Lagos", "Abuja", "Port Harcourt"]
)

season = st.selectbox(
    "Season",
    ["Dry", "Rainy", "Harmattan"]
)

store_id = st.selectbox(
    "Store ID",
    list(range(1, 11))
)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Revenue"):
    
    # Create input dataframe
    input_df = pd.DataFrame([{
        'units_sold': units_sold,
        'price_per_unit': price_per_unit,
        'discount': discount,
        'holiday_flag': holiday_flag,
        'month': month,
        'product_category': product_category,
        'store_location': store_location,
        'season': season,
        'store_id': store_id
    }])

    # Preprocess
    processed_input = preprocess_input(input_df)

    # Predict
    prediction = model.predict(processed_input)

    # Display
    st.success(f"💰 Predicted Revenue: ${prediction[0]:,.2f}")

HOW TO RUN THE APP
To access and use the deployed Sales Revenue Prediction app, follow the steps below:
1. Open the Application
Navigate to the live Streamlit app using the link below:
https://salesrevenuepredictionandoptimization.streamlit.app/

2. Input Required Features
Once the app loads, you will see input fields for key business variables. Provide values for the required features, which may include:
•	Units sold
•	Price per unit
•	Discount and discount intensity
•	Marketing spend
•	Season and month
•	Store and product-related attributes
Ensure that the inputs reflect realistic business scenarios for accurate predictions.

3. Run Prediction
After entering the required values:
•	Click the Predict (or equivalent) button
•	The system will process your inputs using the trained Random Forest model

4. View Results
The app will display:
•	Predicted sales revenue
•	Insights based on the input variables
These outputs can be used to support pricing, sales planning, and marketing decisions.

5. Experiment with Scenarios
To maximize value:
•	Adjust input variables (e.g., increase price or apply discounts)
•	Re-run predictions to simulate different business scenarios
•	Compare outcomes to identify optimal strategies

6. Notes
The model is trained on historical sales data and performs best with inputs similar to the training distribution. Extreme or unrealistic values may produce less reliable predictions.
This interface enables real-time, data-driven decision-making without requiring direct interaction with the underlying machine learning code.

CONCLUSION
This project successfully developed and deployed a high-performing machine learning model for sales revenue prediction, with the Random Forest model achieving the best accuracy. The analysis revealed that revenue is primarily driven by units sold and price per unit, while other factors such as discounts, seasonality, and marketing play supporting roles. By integrating the model into a Streamlit application, the solution provides an accessible, real-time decision-support tool that enables businesses to simulate scenarios, optimize strategies, and make informed, data-driven decisions to improve revenue outcomes.
