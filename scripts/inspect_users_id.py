from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

# Initialize Spark session
spark = SparkSession.builder.appName("MapUserIDs").getOrCreate()

# Correct data path
data_path = "../data/Coursera.csv"
ratings_df = spark.read.csv(data_path, header=True, inferSchema=True)

# Create a unique numerical ID for each user name
user_df = ratings_df.select("partner").distinct().withColumn("user_id", monotonically_increasing_id())
ratings_df = ratings_df.join(user_df, on="partner", how="inner")

# Show the schema and a sample of the data
ratings_df.printSchema()
ratings_df.show(5)

# Save the updated DataFrame if needed
# updated_data_path = "../data/Updated_Coursera.csv"
# ratings_df.write.csv(updated_data_path, header=True)

# Stop Spark session
spark.stop()

