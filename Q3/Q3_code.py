import os
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
import re



spark = SparkSession.builder \
    .master("local[10]") \
    .appName("Q3") \
    .config("spark.driver.memory", "20g") \
    .config("spark.local.dir", os.environ['TMPDIR']) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

import warnings
warnings.filterwarnings("ignore")

def get_param(hyperparams):
    hyper_list = []
    for param, value in hyperparams.items():
        # Extract the name of the hyperparameter.
        hyper_name = param.name
        hyper_list.append({hyper_name: value})
    return hyper_list


print('===================Task A========================')
#Load data
data = spark.read.csv('/users/acp23pks/com6012/ScalableML/Data/HIGGS.csv')
# Rename columns
features = ['label','lepton_pT','lepton_eta','lepton_phi', 'missing_energy_magnitude','missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_btag', 'jet_3_pt', 'jet_3_eta','jet_3_phi', 'jet_3_btag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_btag', 'mjj', 'mjjj', 'mlv', 'mjlv', 'mbb', 'mwbb', 'mwwbb']
ncolumns = len(data.columns)
schemaNames = data.schema.names
#Reassign names to each column
for i in range(ncolumns):
    data = data.withColumnRenamed(schemaNames[i], features[i])
    

# get the names of string columns   
StrColumns = [x.name for x in data.schema.fields if x.dataType == StringType()]
# Convert string columns to double 
for c in StrColumns:
    data = data.withColumn(c, col(c).cast("double"))
    
# Balance classes   
#count the number of positive labels
count_pos = data.filter(data.label==1).count()
# count the number of negative labels
count_neg = data.filter(data.label==0).count()
# calculate the fraction of the minority class
minority_fraction = min(count_pos, count_neg) / float(data.count())
# define class weights based on the minority class
class_weights = {0: minority_fraction, 1: minority_fraction}
# Balance classes
# Calculate the balance ratio
balance_ratio = count_neg / float(count_pos)
# Sample the majority class to balance the dataset
data = data.sampleBy("label", fractions={0: balance_ratio, 1: 1.0}, seed=12)


# Sample and split data
sampled_data = data.sample(False, 0.01, seed=12).cache()
(sam_train_subset, sam_test_subset) = sampled_data.randomSplit([0.7, 0.3], seed=12)

#Write the training and test sets to disk
sam_train_subset.write.mode("overwrite").parquet('/users/acp23pks/com6012/ScalableML/Data/Q1subset_training.parquet')
sam_test_subset.write.mode("overwrite").parquet('/users/acp23pks/com6012/ScalableML/Data/Q1subset_test.parquet')
#Load the training and test sets from disk
subset_train = spark.read.parquet('/users/acp23pks/com6012/ScalableML/Data/Q1subset_training.parquet')
subset_test = spark.read.parquet('/users/acp23pks/com6012/ScalableML/Data/Q1subset_test.parquet')


print('===================Random Forest=========================')
#merge all features to one col
assembler = VectorAssembler(inputCols = features[1:], outputCol = 'features') 
#Creating a Random Forest classifier 
RF = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, impurity='entropy')
#create pipeline
RF_stages = [assembler, RF]
RF_pipeline = Pipeline(stages=RF_stages)

#create a parameter grid    
RF_paramGrid = ParamGridBuilder() \
    .addGrid(RF.maxDepth, [1, 5, 10]) \
    .addGrid(RF.maxBins, [10, 20, 50]) \
    .addGrid(RF.numTrees, [1, 5, 10]) \
    .build()
#cross validation    
RF_crossvalidation = CrossValidator(estimator=RF_pipeline,
                          estimatorParamMaps=RF_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
#Fitting the cross-validator to get best model
RF_cvModel = RF_crossvalidation.fit(subset_train)
#use best model to predict
RF_predictions = RF_cvModel.transform(subset_test)

#Creating an acc evaluator
Acc_evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
#Creating an AUC evaluator              
Area_evaluator = BinaryClassificationEvaluator\
      (labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
      
# calculate accuracy of the predictions      
RF_accuracy = Acc_evaluator.evaluate(RF_predictions)
print("RF accuracy = %g " % RF_accuracy)
# calculate AUC of the predictions
RF_area = Area_evaluator.evaluate(RF_predictions)
print("RF area under the curve = %g " % RF_area)




print('===================Gradient Boosting=========================')
# Create a GBT classifier
GBT = GBTClassifier(maxIter=5, maxDepth=2, labelCol="label", seed=12,
    featuresCol="features", lossType='logistic')

# Define the stages in the GBT pipeline
GBT_stages = [assembler, GBT]
GBT_pipeline = Pipeline(stages=GBT_stages)

#create parameter grid    
GBT_paramGrid = ParamGridBuilder() \
    .addGrid(GBT.maxDepth, [1, 5, 10]) \
    .addGrid(GBT.maxIter, [10, 20, 30]) \
    .addGrid(GBT.stepSize, [0.1, 0.2, 0.05]) \
    .build()
#cross validation    
GBT_crossvalidation = CrossValidator(estimator=GBT_pipeline,
                          estimatorParamMaps=GBT_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
#train model with cross validation
GBT_cvModel = GBT_crossvalidation.fit(subset_train)
GBT_predictions = GBT_cvModel.transform(subset_test)

#calculate GBT acc
GBT_accuracy = Acc_evaluator.evaluate(GBT_predictions)
print("GBT accuracy = %g " % GBT_accuracy)
#calculate GBT AUC
GBT_area = Area_evaluator.evaluate(GBT_predictions)
print("GBT area under the curve = %g " % GBT_area)

print('===================Neural Network=========================')
num_features = len(features) - 1
# Define the layers for the Neural Network. 
# Input layer of size num_features, two intermediary layers, and output of size 2 (for binary classification)
layers = [num_features, 5, 4, 2]

# Create the trainer and set its parameters
NN = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=23766)

# Define the stages in the NN pipeline
NN_stages = [assembler, NN]
NN_pipeline = Pipeline(stages=NN_stages)

# Create the parameter grid for cross-validation
NN_paramGrid = ParamGridBuilder() \
    .addGrid(NN.maxIter, [100, 150]) \
    .addGrid(NN.blockSize, [128, 256]) \
    .addGrid(NN.layers, [[num_features, 5, 4, 2], [num_features, 6, 5, 2]]) \
    .build()

# CrossValidator for the Neural Network
NN_crossvalidation = CrossValidator(estimator=NN_pipeline,
                                    estimatorParamMaps=NN_paramGrid,
                                    evaluator=MulticlassClassificationEvaluator(),
                                    numFolds=5)

# Fitting the cross-validator to get the best model
NN_cvModel = NN_crossvalidation.fit(subset_train)

# Use the best model to predict
NN_predictions = NN_cvModel.transform(subset_test)

# Evaluate the predictions
NN_accuracy = Acc_evaluator.evaluate(NN_predictions)
NN_auc = Area_evaluator.evaluate(NN_predictions)

# Print results
print("Neural Network accuracy = %g " % NN_accuracy)
print("Neural Network area under the curve = %g " % NN_auc)

print('===========================Task B============================')

# Extract the best hyperparameters for the Random Forest model      
RF_hyper = RF_cvModel.getEstimatorParamMaps()[np.argmax(RF_cvModel.avgMetrics)]
RF_params = get_param(RF_hyper)   
print('===================Best parameter for RF========================')  
print(RF_params) 

# Extract the best hyperparameters for the Gradient Boosted Tree model
GBT_hyper = GBT_cvModel.getEstimatorParamMaps()[np.argmax(GBT_cvModel.avgMetrics)]
GBT_params = get_param(GBT_hyper)     
print('=====================Best parameter for GBT======================') 
print(GBT_params)

# Extract the best hyperparameters for the Neural Network model
NN_hyper = NN_cvModel.getEstimatorParamMaps()[np.argmax(NN_cvModel.avgMetrics)]
NN_params = get_param(NN_hyper) 
print('===================Best parameter for NN========================')  
print(NN_params) 

# Read the full training and test data from disk again to get fresh DataFrames without the 'features' column
train = spark.read.parquet('/users/acp23pks/com6012/ScalableML/Data/Q1training.parquet')
test = spark.read.parquet('/users/acp23pks/com6012/ScalableML/Data/Q1test.parquet')

# Random Forest
RF_best = RF_cvModel.bestModel

# No need to use VectorAssembler here again, as RF_best already includes it
RF_predictions = RF_best.transform(test)

# Evaluate Random Forest predictions
RF_accuracy = Acc_evaluator.evaluate(RF_predictions)
RF_auc = Area_evaluator.evaluate(RF_predictions)

print("Random Forest accuracy = %g " % RF_accuracy)
print("Random Forest area under the curve = %g " % RF_auc)
print('===========================================') 

# Gradient Boosting Trees
GBT_best = GBT_cvModel.bestModel

# Similarly, no need for re-assembly since GBT_best includes the VectorAssembler
GBT_predictions = GBT_best.transform(test)

# Evaluate GBT predictions
GBT_accuracy = Acc_evaluator.evaluate(GBT_predictions)
GBT_auc = Area_evaluator.evaluate(GBT_predictions)

print("Gradient Boosted Trees accuracy = %g " % GBT_accuracy)
print("Gradient Boosted Trees area under the curve = %g " % GBT_auc)
print('===========================================') 

# Neural Network
NN_best = NN_cvModel.bestModel

# Similarly, no need for re-assembly since NN_best includes the VectorAssembler
NN_predictions = NN_best.transform(test)

# Evaluate NN predictions
NN_accuracy = Acc_evaluator.evaluate(NN_predictions)
NN_auc = Area_evaluator.evaluate(NN_predictions)

print("Neural Network accuracy = %g " % NN_accuracy)
print("Neural Network area under the curve = %g " % NN_auc)
print('===========================================') 
