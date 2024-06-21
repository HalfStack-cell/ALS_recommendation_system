import os
import logging
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("ALSRecommendationTraining").getOrCreate()

# Load and prepare data
data_path = os.path.join(os.getcwd(), "../data/synthetic_coursera.csv")
logger.info("Loading data from path: %s", data_path)

try:
    data = spark.read.csv(data_path, header=True, inferSchema=True)
except Exception as e:
    logger.error("Error loading data: %s", e)
    spark.stop()
    raise e

# Assemble ALS model
als = ALS(userCol="user_id", itemCol="course_id", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)

# Define a parameter grid for tuning
param_grid = (ParamGridBuilder()
              .addGrid(als.rank, [10, 20, 30, 40, 50])
              .addGrid(als.maxIter, [10, 20, 30, 40])
              .addGrid(als.regParam, [0.01, 0.05, 0.1, 0.2, 0.3])
              .build())

# Define the evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Define CrossValidator
crossval = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Fit the model
logger.info("Starting model tuning with cross-validation")
cv_model = crossval.fit(train_data)

# Get the best model
best_model = cv_model.bestModel

# Save the best model
model_path = os.path.join(os.getcwd(), "models/als_model")
best_model.save(model_path)
logger.info("Best model saved to path: %s", model_path)

# Stop Spark session
spark.stop()
logger.info("Spark session stopped.")








