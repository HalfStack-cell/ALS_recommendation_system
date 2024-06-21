import logging
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("ALSInference").getOrCreate()

# Correct model path
model_path = os.path.join(os.getcwd(), "models/als_model")
logger.info("Loading ALS model from path: %s", model_path)

try:
    als_model = ALSModel.load(model_path)
except Exception as e:
    logger.error("Error loading ALS model: %s", e)
    spark.stop()
    raise e

# Correct data path
data_path = os.path.join(os.getcwd(), "../data/synthetic_coursera.csv")
logger.info("Loading test data from path: %s", data_path)

try:
    ratings_df = spark.read.csv(data_path, header=True, inferSchema=True)
except Exception as e:
    logger.error("Error loading test data: %s", e)
    spark.stop()
    raise e

# Print the schema of the DataFrame
print("Schema of the DataFrame:")
ratings_df.printSchema()

# Check the first few rows
ratings_df.show(5)

# Prepare test data
ratings_df = ratings_df.dropna(subset=["user_id", "course_id", "rating"])
ratings_df = ratings_df.withColumn("user_id", ratings_df["user_id"].cast("int"))
ratings_df = ratings_df.withColumn("course_id", ratings_df["course_id"].cast("int"))
ratings_df = ratings_df.withColumn("rating", ratings_df["rating"].cast("float"))

# Sample user IDs to generate recommendations for
sample_user_ids = [1, 2, 3]  # Replace with valid user_ids from the dataset

try:
    for sample_user_id in sample_user_ids:
        logger.info(f"Generating recommendations for user_id: {sample_user_id}")
        user_ratings = ratings_df.filter(ratings_df.user_id == sample_user_id)
        user_recommendations = als_model.recommendForUserSubset(user_ratings, 5)
        recommendations = user_recommendations.collect()

        # Print recommendations in a readable format
        for row in recommendations:
            print(f"Recommendations for user_id {row.user_id}:")
            for rec in row.recommendations:
                print(f"Course ID: {rec.course_id}, Rating: {rec.rating:.2f}")

except Exception as e:
    logger.error("Error during model inference: %s", e)

finally:
    spark.stop()
    logger.info("Spark session stopped.")
    logger.info("Closing down clientserver connection")





# The code snippet above loads a saved ALS model and uses it to generate recommendations for a sample user.
