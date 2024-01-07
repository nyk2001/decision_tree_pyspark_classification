# Databricks notebook source
# MAGIC %md
# MAGIC ## Read Dataset 
# Magic 
# COMMAND ----------

import pandas as pd

# Read data into pandas data frame because Databricks can only read from local file
# Not ideal solution but works for demo
cancer_pd =pd.read_csv(os.path.join(os.getcwd(),"cancer_dataset.csv"))
cancer_df=spark.createDataFrame(cancer_pd) 

print(f"Total Rows = {cancer_df.count()}")

# COMMAND ----------

display(cancer_df)

# COMMAND ----------

# Display schema
for x in cancer_df.schema:
    print(f"File name : {x.name}  |   Data Type  : {x.dataType}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Dropping columns

# COMMAND ----------

# Remove id column and the last column 
print(f"Total columns before  = {len(cancer_df.columns)}")
cancer_df= cancer_df.drop("id", "Unnamed: 32")
print(f"Total columns after   = {len(cancer_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Change columns data types

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col

# Change data types of all columns to double except the diagnosis column
# diagnosis is the label column
for x in cancer_df.schema:
    if x.name!="diagnosis":
        if x.dataType == StringType():
            cancer_df = cancer_df.withColumn(x.name, col(x.name).castTo(DoubleType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create String indexer and Vector assembler

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler

# split dataset 
train_df, test_df = cancer_df.randomSplit([.8,.2], seed = 10)

# convert label to integeres - LabelEncoder in sklearn
string_indexer = StringIndexer(inputCol = "diagnosis", outputCol ="label")

# create vector indexer for mean features only - ten real valued features
numerical_cols = [x.name for x in train_df.schema if ((x.name!="diagnosis") & (x.name!="label") & ("_mean" in x.name))]
vector_assembler = VectorAssembler(inputCols =numerical_cols , outputCol ="features")


# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create pipeline and fit decision tree

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

# create a decision tree
dtree = DecisionTreeClassifier(featuresCol = "features", labelCol = "label")

# create pipeline
stages= [string_indexer, vector_assembler, dtree ]
pipeline = Pipeline(stages = stages)

# fit pipeline - model
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# display the tree
dt_model = pipeline_model.stages[-1]
display(dt_model)

# COMMAND ----------

# display feature importance
features_df = pd.DataFrame(list(zip(vector_assembler.getInputCols(), dt_model.featureImportances)), columns=["feature", "importance"])
features_df.sort_values("importance", ascending =False)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Generate predictions on test dataset

# COMMAND ----------

# transform the test data set
results_df  = pipeline_model.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ######Binary Classifier metrics

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Tranform test data
output_df = pipeline_model.transform(test_df)

# Generate metrics
bce =  BinaryClassificationEvaluator(rawPredictionCol = "prediction" , labelCol= "label", metricName= "areaUnderPR")
print(f"Area Under PR = {bce.evaluate(output_df)}")
bce.setMetricName("areaUnderROC")
print(f"Area Under ROC = {bce.evaluate(output_df)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ######Multiclass Classification Evaluation metrics

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

bce =  MulticlassClassificationEvaluator(predictionCol = "prediction" , labelCol= "label", metricName= "accuracy")
print(f"Accuracy = {bce.evaluate(output_df)}")
bce.setMetricName("f1")
print(f"F1 score = {bce.evaluate(output_df)}")
bce.setMetricName("precisionByLabel")
print(f"Precision = {bce.evaluate(output_df)}")
bce.setMetricName("recallByLabel")
print(f"Recall = {bce.evaluate(output_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ######Confuison matrix

# COMMAND ----------

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import DoubleType
import seaborn as sns
import matplotlib.pyplot as plt     

# stats about labels of test data set 
print(f"Total Malignant = {test_df.filter(col('diagnosis')=='M').count()}")
print(f"Total Bening = {test_df.filter(col('diagnosis')=='B').count()}")

# Compute raw scores on the test set
predictionAndLabels = output_df[["prediction", "label"]]
predictionAndLabels= predictionAndLabels.withColumn("prediction", col("prediction").cast(DoubleType()))
predictionAndLabels= predictionAndLabels.withColumn("label", col("label").cast(DoubleType()))

# instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels.rdd)

# generate a confusion matrix
confusion_matrix =  metrics.confusionMatrix().toArray()

# plot the matrix
ax= plt.subplot()
sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax, cmap= "Blues");  

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Benign', 'Malignant']); ax.yaxis.set_ticklabels(['Benign', 'Malignant']);

# COMMAND ----------


