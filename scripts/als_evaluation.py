import logging
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("ALSModelEvaluation").getOrCreate()

# Correct model path
model_path = os.path.join(os.getcwd(), "models/als_model")
logger.info("Loading ALS model from path: %s", model_path)

# Correct data path
data_path = os.path.join(os.getcwd(), "../data/synthetic_coursera.csv")
logger.info("Loading test data from path: %s", data_path)

try:
    # Load and prepare test data
    test_data = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.info("Schema of the DataFrame:")
    test_data.printSchema()
except Exception as e:
    logger.error("Error loading test data: %s", e)
    spark.stop()
    raise e

# Define ALS estimator
als = ALS(userCol="user_id", itemCol="course_id", ratingCol="rating", coldStartStrategy="drop")

# Define parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 20, 30]) \
    .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(als.maxIter, [10, 20]) \
    .build()

# Define evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Define cross-validator
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# Train and evaluate model
try:
    cvModel = crossval.fit(test_data)
    bestModel = cvModel.bestModel

    # Print best parameters
    logger.info("Best Rank: %s", bestModel.rank)
    logger.info("Best Regularization Parameter: %s", bestModel._java_obj.parent().getRegParam())
    logger.info("Best Max Iterations: %s", bestModel._java_obj.parent().getMaxIter())

    # Make predictions
    predictions = bestModel.transform(test_data)

    # Evaluate the model
    rmse = evaluator.evaluate(predictions)
    logger.info("Root-mean-square error = %s", rmse)

    # Show predictions
    predictions.select("user_id", "course_id", "rating", "prediction").show(10)
except Exception as e:
    logger.error("Error during model training and evaluation: %s", e)
    spark.stop()
    raise e

# Stop Spark session
spark.stop()
logger.info("Spark session stopped.")
logger.info("Closing down clientserver connection")









