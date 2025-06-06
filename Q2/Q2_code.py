import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StandardScaler, StringIndexer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import expr

# Set up the Spark session
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Q2") \
    .config("spark.local.dir", "/mnt/parscratch/users/acp23pks") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

# Load and preprocess data
data = spark.read.csv('/users/acp23pks/com6012/ScalableML/Data/freMTPL2freq.csv', header=True, inferSchema=True)
data.cache()
data.show(40, False)

# Task A: Pre-processing
# a. Create a new column hasClaim
data = data.withColumn("hasClaim", when(col("ClaimNb") > 0, 1).otherwise(0))

# b. Stratified split into training and test sets
fractions = {0: 0.7, 1: 0.3}  # 70% for training, 30% for test for each class
train = data.sampleBy("hasClaim", fractions, seed=23766)
test = data.subtract(train)

# Task B: Training predictive models
# Define the feature columns to be used
feature_cols = ['Exposure', 'Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region']

# Identify categorical columns which need one-hot encoding
categoricalCols = [col for col in feature_cols if data.schema[col].dataType == StringType()]

# Define StringIndexer and OneHotEncoder for categorical features
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categoricalCols]
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=f"{indexer.getOutputCol()}_ohe") for indexer in indexers]

# Identify numeric columns, which do not need one-hot encoding
numericCols = [col for col in feature_cols if col not in categoricalCols]

# Assemble input columns for VectorAssembler
assemblerInputs = [encoder.getOutputCol() for encoder in encoders] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# Define StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Define regression models
glm_poisson = GeneralizedLinearRegression(family="poisson", link="log", labelCol="ClaimNb")
lr_l1 = LogisticRegression(labelCol="hasClaim", elasticNetParam=1)  # L1 regularization
lr_l2 = LogisticRegression(labelCol="hasClaim", elasticNetParam=0)  # L2 regularization

# Define pipelines for each model
pipeline_poisson = Pipeline(stages=indexers + encoders + [assembler, scaler, glm_poisson])
pipeline_lr_l1 = Pipeline(stages=indexers + encoders + [assembler, scaler, lr_l1])
pipeline_lr_l2 = Pipeline(stages=indexers + encoders + [assembler, scaler, lr_l2])

# Define the parameter grid for cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(glm_poisson.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .addGrid(lr_l1.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .addGrid(lr_l2.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()

# Set up CrossValidator for the Poisson model
crossval_poisson = CrossValidator(estimator=pipeline_poisson,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=RegressionEvaluator(labelCol="ClaimNb"),
                                  numFolds=3)
# Fit the Poisson model
cvModel_poisson = crossval_poisson.fit(train.sample(False, 0.1, seed=23766))

# Set up CrossValidators for Logistic Regression models
crossval_lr_l1 = CrossValidator(estimator=pipeline_lr_l1,
                                estimatorParamMaps=paramGrid,
                                evaluator=BinaryClassificationEvaluator(labelCol="hasClaim"),
                                numFolds=3)
crossval_lr_l2 = CrossValidator(estimator=pipeline_lr_l2,
                                estimatorParamMaps=paramGrid,
                                evaluator=BinaryClassificationEvaluator(labelCol="hasClaim"),
                                numFolds=3)
# Fit the Logistic Regression models
cvModel_lr_l1 = crossval_lr_l1.fit(train.sample(False, 0.1, seed=23766))
cvModel_lr_l2 = crossval_lr_l2.fit(train.sample(False, 0.1, seed=23766))

# Task B.b. Utilize the optimal hyperparameters, and train your models on the full dataset
bestModel_poisson = cvModel_poisson.bestModel
predictions_poisson = bestModel_poisson.transform(test)

bestModel_lr_l1 = cvModel_lr_l1.bestModel
predictions_lr_l1 = bestModel_lr_l1.transform(test)

bestModel_lr_l2 = cvModel_lr_l2.bestModel
predictions_lr_l2 = bestModel_lr_l2.transform(test)

# Evaluate the models
evaluator_poisson = RegressionEvaluator(labelCol="ClaimNb", metricName="rmse")
rmse_poisson = evaluator_poisson.evaluate(predictions_poisson)

evaluator_lr = BinaryClassificationEvaluator(labelCol="hasClaim", metricName="areaUnderROC")
auc_lr_l1 = evaluator_lr.evaluate(predictions_lr_l1)
auc_lr_l2 = evaluator_lr.evaluate(predictions_lr_l2)

# Accuracy calculation for Logistic Regression with L1 Regularization
accuracy_lr_l1 = predictions_lr_l1.withColumn('correct', expr("float(prediction = hasClaim)")).selectExpr("AVG(correct)").first()[0]
print("Logistic Regression Accuracy (L1): ", accuracy_lr_l1)


# Accuracy calculation for Logistic Regression with L2 Regularization
accuracy_lr_l2 = predictions_lr_l2.withColumn('correct', expr("float(prediction = hasClaim)")).selectExpr("AVG(correct)").first()[0]
print("Logistic Regression Accuracy (L2): ", accuracy_lr_l2)


# Print the results
print("Poisson RMSE: ", rmse_poisson)
print("Logistic Regression AUC (L1): ", auc_lr_l1)
print("Logistic Regression AUC (L2): ", auc_lr_l2)

# Print model coefficients
print("Poisson Model Coefficients: ", bestModel_poisson.stages[-1].coefficients)
print("Logistic Regression Coefficients (L1): ", bestModel_lr_l1.stages[-1].coefficients)
print("Logistic Regression Coefficients (L2): ", bestModel_lr_l2.stages[-1].coefficients)

# Shutdown Spark session
spark.stop()

