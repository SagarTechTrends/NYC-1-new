# Databricks notebook source
file_location = "dbfs:/FileStore/tables/nyc_rolling_sales.csv"
file_type = "csv"
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)


# COMMAND ----------

from pyspark.sql.functions import col, to_date, year, month, udf, regexp_replace
from pyspark.sql.types import StringType

# Clean numeric columns first
cols_to_clean = ["SALE PRICE", "GROSS SQUARE FEET", "LAND SQUARE FEET"]
for c in cols_to_clean:
    df = df.withColumn(c, regexp_replace(col(c), ",", ""))
    df = df.withColumn(c, col(c).cast("double"))

# Filter non-zero and valid sales
df = df.filter((col("SALE PRICE").isNotNull()) & (col("SALE PRICE") > 0))

# Convert SALE DATE
df = df.withColumn("SALE DATE", to_date(col("SALE DATE"), "MM/dd/yyyy"))
df = df.withColumn("SALE YEAR", year(col("SALE DATE")))
df = df.withColumn("SALE MONTH", month(col("SALE DATE")))

# Add SEASON
def get_season(m):
    return (
        "Winter" if m in [12, 1, 2] else
        "Spring" if m in [3, 4, 5] else
        "Summer" if m in [6, 7, 8] else
        "Fall"
    )

season_udf = udf(get_season, StringType())
df = df.withColumn("SEASON", season_udf(col("SALE MONTH")))

# Drop rows with missing values in key columns
df = df.dropna(subset=[
    "SALE PRICE", "SALE DATE", "YEAR BUILT", 
    "GROSS SQUARE FEET", "LAND SQUARE FEET"
])

display(df)


# COMMAND ----------

from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Create new numerical features
df = df.withColumn("BUILDING AGE", col("SALE YEAR") - col("YEAR BUILT"))
df = df.withColumn("PRICE PER SQFT", col("SALE PRICE") / col("GROSS SQUARE FEET"))

# Create UNIT_BIN
df = df.withColumn("UNIT_BIN", 
    when(col("TOTAL UNITS") <= 1, "1")
    .when(col("TOTAL UNITS") <= 5, "2–5")
    .when(col("TOTAL UNITS") <= 10, "6–10")
    .otherwise("11+")
)

# Categorical columns to encode
index_cols = [
    "BOROUGH", "BUILDING CLASS CATEGORY", "SEASON",
    "TAX CLASS AT PRESENT", "TAX CLASS AT TIME OF SALE",
    "BUILDING CLASS AT PRESENT", "BUILDING CLASS AT TIME OF SALE",
    "UNIT_BIN", "SALE MONTH", "SALE YEAR"
]

indexed = [c + "_IDX" for c in index_cols]
encoded = [c + "_ENC" for c in index_cols]

# Build encoding pipeline
indexers = [StringIndexer(inputCol=c, outputCol=i, handleInvalid='keep') for c, i in zip(index_cols, indexed)]
encoders = [OneHotEncoder(inputCol=i, outputCol=o) for i, o in zip(indexed, encoded)]

pipeline = Pipeline(stages=indexers + encoders)
df = pipeline.fit(df).transform(df)

display(df.select("BUILDING AGE", "PRICE PER SQFT", "UNIT_BIN", *encoded))


# COMMAND ----------

q1 = df.approxQuantile("SALE PRICE", [0.25], 0.0)[0]
q3 = df.approxQuantile("SALE PRICE", [0.75], 0.0)[0]
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df = df.filter((col("SALE PRICE") >= lower_bound) & (col("SALE PRICE") <= upper_bound))

display(df.select("SALE PRICE"))


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

df_plot = df.select(
    "SALE PRICE", "GROSS SQUARE FEET", "LAND SQUARE FEET",
    "BUILDING AGE", "PRICE PER SQFT",
    "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "TOTAL UNITS"
).toPandas()

plt.figure(figsize=(10, 6))
sns.heatmap(df_plot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# COMMAND ----------

df_borough = df.select("BOROUGH", "SALE PRICE").toPandas()
df_borough["SALE PRICE"] = df_borough["SALE PRICE"].astype(float)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_borough, x="BOROUGH", y="SALE PRICE")
plt.title("Sale Price Distribution by Borough")
plt.xlabel("Borough")
plt.ylabel("Sale Price")
plt.show()

# COMMAND ----------

from pyspark.sql.functions import when
df.groupBy("UNIT_BIN").avg("SALE PRICE").orderBy("UNIT_BIN").show()
df.groupBy("SEASON").avg("SALE PRICE").orderBy("SEASON").show()
df.groupBy("BOROUGH").avg("PRICE PER SQFT").orderBy("BOROUGH").show()
df.groupBy("SEASON").avg("PRICE PER SQFT").orderBy("SEASON").show()
df.groupBy("BUILDING CLASS CATEGORY").avg("SALE PRICE").orderBy("avg(SALE PRICE)", ascending=False).show(10)

# COMMAND ----------

df_season_count = df.groupBy("SEASON").count().toPandas()
sns.barplot(data=df_season_count, x="SEASON", y="count")
plt.title("Number of Sales by Season")
plt.ylabel("Number of Properties Sold")
plt.show()


# COMMAND ----------

df_neigh = df.groupBy("NEIGHBORHOOD").avg("SALE PRICE") \
    .withColumnRenamed("avg(SALE PRICE)", "AVG_PRICE") \
    .orderBy(col("AVG_PRICE").desc()).limit(10).toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_neigh, x="NEIGHBORHOOD", y="AVG_PRICE")
plt.xticks(rotation=45)
plt.title("Top 10 Neighborhoods by Average Sale Price")
plt.ylabel("Average Sale Price")
plt.show()


# COMMAND ----------

df_zip = df.groupBy("ZIP CODE").avg("SALE PRICE") \
    .withColumnRenamed("avg(SALE PRICE)", "AVG_PRICE") \
    .orderBy(col("AVG_PRICE").desc()).limit(10).toPandas()

plt.figure(figsize=(10, 5))
sns.barplot(data=df_zip, x="ZIP CODE", y="AVG_PRICE")
plt.xticks(rotation=45)
plt.title("Top 10 ZIP Codes by Average Sale Price")
plt.ylabel("Average Sale Price")
plt.show()


# COMMAND ----------

from pyspark.sql.functions import col
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Numerical-only features
numerical_features = [
    "BUILDING AGE", "PRICE PER SQFT",
    "GROSS SQUARE FEET", "LAND SQUARE FEET",
    "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "TOTAL UNITS"
]

# Convert only numerical data to Pandas
df_vif = df.select(*numerical_features).dropna().toPandas()

# Compute VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
vif_data = vif_data.sort_values(by="VIF", ascending=False)

vif_data


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col

# Step 1: Define Categorical Columns
index_cols = [
    "BOROUGH", "BUILDING CLASS CATEGORY", "SEASON",
    "TAX CLASS AT PRESENT", "TAX CLASS AT TIME OF SALE",
    "BUILDING CLASS AT PRESENT", "BUILDING CLASS AT TIME OF SALE",
    "UNIT_BIN", "SALE MONTH", "SALE YEAR"
]

# Step 2: Generate Indexed and Encoded Column Names
indexed = [c + "_IDX" for c in index_cols]
encoded = [c + "_ENC" for c in index_cols]

# Step 3: Drop if columns already exist to avoid overwrite errors
cols_to_drop = [c for pair in zip(indexed, encoded) for c in pair if c in df.columns]
df = df.drop(*cols_to_drop)

# Step 4: Create StringIndexers and OneHotEncoders
indexers = [StringIndexer(inputCol=c, outputCol=i, handleInvalid="keep") for c, i in zip(index_cols, indexed)]
encoders = [OneHotEncoder(inputCol=i, outputCol=o) for i, o in zip(indexed, encoded)]

# Step 5: Build and Apply Pipeline
pipeline = Pipeline(stages=indexers + encoders)
df = pipeline.fit(df).transform(df)

# Step 6: Define Final Feature List
numerical = [
    "BUILDING AGE", "PRICE PER SQFT", 
    "GROSS SQUARE FEET", "LAND SQUARE FEET", 
    "RESIDENTIAL UNITS"
]
features = numerical + encoded


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Drop rows with nulls in any feature or label column
df_clean = df.dropna(subset=features + ["SALE PRICE"])

# Assemble feature vector
assembler = VectorAssembler(inputCols=features, outputCol="features_vec")
df_model = assembler.transform(df_clean).select("features_vec", col("SALE PRICE").alias("label"))

# Train-test split
train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features_vec", labelCol="label")
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

print(f"Intercept: {lr_model.intercept}")
print(f"Number of Coefficients: {len(lr_model.coefficients)}")


# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# Initialize Random Forest model
rf = RandomForestRegressor(featuresCol="features_vec", labelCol="label", numTrees=100)

# Train the model
rf_model = rf.fit(train_data)

# Make predictions
rf_preds = rf_model.transform(test_data)

# Optional: Show sample predictions
rf_preds.select("prediction", "label").show(5)


# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Convert Spark DataFrames to Pandas
train_pd = train_data.toPandas()
test_pd = test_data.toPandas()

# Unpack feature vectors into 2D numpy arrays
X_train = np.array(train_pd["features_vec"].tolist())
y_train = train_pd["label"]
X_test = np.array(test_pd["features_vec"].tolist())
y_test = test_pd["label"]

# Train the XGBoost Regressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# Predict on the test set
xgb_preds = xgb.predict(X_test)

# Optional: Display a few predictions
for i in range(5):
    print(f"Predicted: {xgb_preds[i]:,.2f}, Actual: {y_test.iloc[i]:,.2f}")


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- Linear Regression Evaluation ---
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse_lr = evaluator_rmse.evaluate(lr_preds)
r2_lr = evaluator_r2.evaluate(lr_preds)

# --- Random Forest Evaluation ---
rmse_rf = evaluator_rmse.evaluate(rf_preds)
r2_rf = evaluator_r2.evaluate(rf_preds)

# --- XGBoost Evaluation (in pandas) ---
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_preds))
r2_xgb = r2_score(y_test, xgb_preds)

# --- Print Comparison Table ---
print(f"{'Model':<20}{'RMSE':<15}{'R²':<15}")
print(f"{'-'*50}")
print(f"{'Linear Regression':<20}{rmse_lr:<15.2f}{r2_lr:<15.4f}")
print(f"{'Random Forest':<20}{rmse_rf:<15.2f}{r2_rf:<15.4f}")
print(f"{'XGBoost':<20}{rmse_xgb:<15.2f}{r2_xgb:<15.4f}")


# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

# Initialize model
xgb = XGBRegressor()

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",  # or use "r2"
    verbose=1,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict using best model
xgb_preds_tuned = best_xgb.predict(X_test)


# COMMAND ----------

from sklearn.metrics import mean_squared_error, r2_score

rmse_tuned = np.sqrt(mean_squared_error(y_test, xgb_preds_tuned))
r2_tuned = r2_score(y_test, xgb_preds_tuned)

print(f"Tuned XGBoost RMSE: {rmse_tuned:,.2f}")
print(f"Tuned XGBoost R²: {r2_tuned:.4f}")


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# PySpark evaluator for LR and RF
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

# --- Linear Regression ---
rmse_lr = evaluator_rmse.evaluate(lr_preds)
r2_lr = evaluator_r2.evaluate(lr_preds)

# --- Random Forest ---
rmse_rf = evaluator_rmse.evaluate(rf_preds)
r2_rf = evaluator_r2.evaluate(rf_preds)

# --- XGBoost (default) ---
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_preds))
r2_xgb = r2_score(y_test, xgb_preds)

# --- XGBoost (tuned) ---
rmse_tuned = np.sqrt(mean_squared_error(y_test, xgb_preds_tuned))
r2_tuned = r2_score(y_test, xgb_preds_tuned)

# --- Display Results ---
print(f"{'Model':<20}{'RMSE':<15}{'R²':<15}")
print(f"{'-'*50}")
print(f"{'Linear Regression':<20}{rmse_lr:<15.2f}{r2_lr:<15.4f}")
print(f"{'Random Forest':<20}{rmse_rf:<15.2f}{r2_rf:<15.4f}")
print(f"{'XGBoost (Default)':<20}{rmse_xgb:<15.2f}{r2_xgb:<15.4f}")
print(f"{'XGBoost (Tuned)':<20}{rmse_tuned:<15.2f}{r2_tuned:<15.4f}")


# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# Step 1: Convert Spark to Pandas
import pandas as pd
df_pandas = df.select(
    "SALE PRICE", "BUILDING AGE", "PRICE PER SQFT", 
    "GROSS SQUARE FEET", "LAND SQUARE FEET", "RESIDENTIAL UNITS",
    "BOROUGH", "BUILDING CLASS CATEGORY", "SEASON", 
    "TAX CLASS AT PRESENT", "TAX CLASS AT TIME OF SALE",
    "BUILDING CLASS AT PRESENT", "BUILDING CLASS AT TIME OF SALE",
    "UNIT_BIN", "SALE MONTH"
).dropna().toPandas()

# Step 2: Encode categoricals with meaningful names
df_pandas = pd.get_dummies(df_pandas, drop_first=True)

# Step 3: Split X, y
X = df_pandas.drop("SALE PRICE", axis=1)
y = df_pandas["SALE PRICE"]

# Step 4: Train new XGBoost model
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X, y)

# Step 5: SHAP with real names
import shap
explainer = shap.Explainer(xgb)
shap_values = explainer(X[:100])

shap.summary_plot(shap_values, X.iloc[:100])
