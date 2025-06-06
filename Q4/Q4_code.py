import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, explode, avg

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Movie Recommendation and Cluster Analysis") \
    .config("spark.local.dir", os.environ['TMPDIR']) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Task A: Time-split recommendation
print('=======================Task A================')

# Load data
data_path = '/users/acp23pks/com6012/ScalableML/Data/ml-20m/ratings.csv'
origin_datas = spark.read.csv(data_path, header=True, inferSchema=True).cache()

# Sort all data by the timestamp
data_sorted = origin_datas.orderBy('timestamp', ascending=True).cache()

# Calculating percentage rank for splitting
windowSpec = Window.orderBy('timestamp')
data_ranks = data_sorted.withColumn("percent_rank", F.percent_rank().over(windowSpec))
data_ranks.show(40, False)

# Splitting the data
train1 = data_ranks.filter(data_ranks["percent_rank"] < 0.4).cache()
test1 = data_ranks.filter(data_ranks["percent_rank"] >= 0.4).cache()

train2 = data_ranks.filter(data_ranks["percent_rank"] < 0.6).cache()
test2 = data_ranks.filter(data_ranks["percent_rank"] >= 0.6).cache()

train3 = data_ranks.filter(data_ranks["percent_rank"] < 0.8).cache()
test3 = data_ranks.filter(data_ranks["percent_rank"] >= 0.8).cache()

# ALS configurations
seed_value = 230123766  # Assuming this is the student number seed
als1 = ALS(userCol="userId", itemCol="movieId", seed=seed_value, coldStartStrategy="drop")

# Define the evaluators
rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

# Function to run ALS model and compute metrics
def run_model(train, test, als_model):
    model = als_model.fit(train)
    predictions = model.transform(test)
    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    return rmse, mse, mae

# Store metrics for visualization
metrics_dict = {'Setting1': {'40%': {}, '60%': {}, '80%': {}},
                'Setting2': {'40%': {}, '60%': {}, '80%': {}}}

# Run models and collect metrics for Setting 1
metrics_dict['Setting1']['40%'] = run_model(train1, test1, als1)
metrics_dict['Setting1']['60%'] = run_model(train2, test2, als1)
metrics_dict['Setting1']['80%'] = run_model(train3, test3, als1)

# Modify ALS setting based on analysis of Setting 1 results for Setting 2
als2 = ALS(userCol="userId", itemCol="movieId", seed=seed_value, coldStartStrategy="drop", rank=20, maxIter=15, regParam=0.01)
metrics_dict['Setting2']['40%'] = run_model(train1, test1, als2)
metrics_dict['Setting2']['60%'] = run_model(train2, test2, als2)
metrics_dict['Setting2']['80%'] = run_model(train3, test3, als2)

# Function to format and display ALS metrics neatly in one DataFrame
def format_als_metrics(metrics_dict):
    rows = []
    for setting, data in metrics_dict.items():
        for split, metrics in data.items():
            rmse, mse, mae = metrics
            rows.append([split, setting, rmse, mse, mae])
    return pd.DataFrame(rows, columns=["Split", "Setting", "RMSE", "MSE", "MAE"])

als_metrics_df = format_als_metrics(metrics_dict)

print("ALS Metrics Table Overview:")
print(tabulate(als_metrics_df, headers='keys', tablefmt='grid'))

# Visualize the plot
plt.figure(figsize=(10, 6))
splits = ['40%', '60%', '80%']
metrics = ['RMSE', 'MSE', 'MAE']
colors = {'RMSE': 'blue', 'MSE': 'green', 'MAE': 'red'}
line_styles = {'Setting1': '-', 'Setting2': '--'}  # Different line styles for different settings

# Iterate through each metric and plot across all splits for both settings
for metric in metrics:
    for setting in ['Setting1', 'Setting2']:
        values = [metrics_dict[setting][split][metrics.index(metric)] for split in splits]
        plt.plot(splits, values, label=f'{metric} ({setting})', color=colors[metric], linestyle=line_styles[setting], marker='o')

        # Add text annotations for each point
        for (i, value) in enumerate(values):
            plt.text(splits[i], value, f'{value:.2f}', color=colors[metric], ha='center', va='bottom')

plt.title('Performance of ALS Settings Across Different Splits')
plt.xlabel('Training Split Percentage')
plt.ylabel('Metric Values')
plt.legend(title="Metrics and Settings")
plt.grid(True)

plt.tight_layout()
plt.savefig('/users/acp23pks/com6012/ScalableML/Output/Q4_figA3.jpg')

# Task B: User Analysis
print('=======================Task B================')
# Function to extract feature vectors and perform k-means clustering
def kmeans_clustering(train_data, k, seed_value):
    model = als2.fit(train_data)
    user_factors = model.userFactors
    kmeans = KMeans(k=k, seed=seed_value, featuresCol='features', predictionCol='cluster')
    clusters = kmeans.fit(user_factors).transform(user_factors)
    return clusters

# Perform clustering for each time-split
clusters1 = kmeans_clustering(train1, 25, seed_value)
clusters2 = kmeans_clustering(train2, 25, seed_value)
clusters3 = kmeans_clustering(train3, 25, seed_value)

# Function to find top 5 largest clusters
def top_clusters(clusters):
    cluster_sizes = clusters.groupBy('cluster').count().orderBy('count', ascending=False).take(5)
    return [row['count'] for row in cluster_sizes]

# Collect cluster sizes for visualization
cluster_sizes = {
    '40%': top_clusters(clusters1),
    '60%': top_clusters(clusters2),
    '80%': top_clusters(clusters3)
}

# Creating a DataFrame for cluster sizes
cluster_sizes_df = pd.DataFrame(cluster_sizes, index=['1st Largest', '2nd Largest', '3rd Largest', '4th Largest', '5th Largest'])
print('Cluster Sizes Table Overview:')
print(tabulate(cluster_sizes_df, headers='keys', tablefmt='grid'))

# Visualize the cluster sizes
fig, ax = plt.subplots(figsize=(10, 6))
splits = ['40%', '60%', '80%']
width = 0.15  # Slightly reduce the width for better visibility
x = range(len(splits))

for i, size in enumerate(['1st', '2nd', '3rd', '4th', '5th']):
    # Position for each cluster bar in each group
    pos = [p + i * width for p in x]
    cluster_data = [cluster_sizes[split][i] for split in splits]
    bars = ax.bar(pos, cluster_data, width, label=f'{size} largest')

    # Add text annotations on each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

ax.set_xlabel('Training Size Split')
ax.set_ylabel('Number of Users in Cluster')
ax.set_title('Top 5 Largest Clusters by Training Size Split')
ax.set_xticks([p + 2 * width for p in x])  # Adjust x-ticks to center labels between the groups
ax.set_xticklabels(splits)
ax.legend(title="Cluster Rank", bbox_to_anchor=(1.05, 1), loc='upper left')  

plt.tight_layout()
plt.savefig('/users/acp23pks/com6012/ScalableML/Output/Q4_figB1.jpg')

# Perform Task B-2
# Function to get the top movies within the largest cluster
def get_movies_largest_cluster(train_data, clusters):
    largest_cluster_id = clusters.groupBy('cluster').count().orderBy('count', ascending=False).first()[0]
    user_ids = clusters.filter(clusters.cluster == largest_cluster_id).select('id').rdd.map(lambda r: r[0]).collect()
    movie_ratings = train_data.filter((train_data.userId.isin(user_ids)) & (train_data.rating >= 4))
    movies_largest_cluster = movie_ratings.groupBy('movieId').agg(F.avg('rating').alias('average_rating'))
    return movies_largest_cluster.filter('average_rating >= 4')

movies_largest_cluster1 = get_movies_largest_cluster(train1, clusters1)
movies_largest_cluster2 = get_movies_largest_cluster(train2, clusters2)
movies_largest_cluster3 = get_movies_largest_cluster(train3, clusters3)

# Load movie genres
movie_data_path = '/users/acp23pks/com6012/ScalableML/Data/ml-20m/movies.csv'
movies = spark.read.csv(movie_data_path, header=True, inferSchema=True).cache()

# Function to find top 10 genres from top movies
def get_top_genres_with_counts(movies_largest_cluster, movies):
    top_movie_ids = movies_largest_cluster.select('movieId').rdd.map(lambda r: r[0]).collect()
    top_movie_genres = movies.filter(movies.movieId.isin(top_movie_ids))
    genre_counts = top_movie_genres.select(explode(F.split(col('genres'), '\|')).alias('genre')).groupBy('genre').count()
    return genre_counts.orderBy('count', ascending=False).limit(10).collect()

top_genres_df1 = get_top_genres_with_counts(movies_largest_cluster1, movies)
top_genres_df2 = get_top_genres_with_counts(movies_largest_cluster2, movies)
top_genres_df3 = get_top_genres_with_counts(movies_largest_cluster3, movies)

# Convert lists of tuples to a DataFrame directly suitable for visualization
def create_genre_df(top_genres_list, split):
    return pd.DataFrame({
        'Split': [split] * len(top_genres_list),
        'Genre': [genre[0] for genre in top_genres_list],
        'Count': [genre[1] for genre in top_genres_list]
    })

# Function to summarize and display top movies and genres
def summarize_top_movies_genres(top_movies, movies_df, split_label):
    # Convert Spark DataFrame to pandas DataFrame
    top_movies_df = top_movies.toPandas()
    if isinstance(movies_df, pd.DataFrame):
        movies_pandas_df = movies_df
    else:
        movies_pandas_df = movies_df.toPandas()

    # Merge dataframes
    merged_df = pd.merge(top_movies_df, movies_pandas_df, on="movieId", how="left")
    # Calculate the count of genres
    genre_counts = merged_df['genres'].str.get_dummies(sep='|').sum()
    genre_counts_df = genre_counts.sort_values(ascending=False).head(10).reset_index()
    genre_counts_df.columns = ['Genre', 'Count']
    genre_counts_df['Split'] = split_label
    return genre_counts_df

# Analyze top movies from each training split
top_genres_df1 = summarize_top_movies_genres(movies_largest_cluster1, movies, "40% Split")
top_genres_df2 = summarize_top_movies_genres(movies_largest_cluster2, movies, "60% Split")
top_genres_df3 = summarize_top_movies_genres(movies_largest_cluster3, movies, "80% Split")

# Combine the DataFrames into one
combined_genres_df = pd.concat([top_genres_df1, top_genres_df2, top_genres_df3], ignore_index=True)

# Optionally, reorder the columns for better readability
combined_genres_df = combined_genres_df[['Split', 'Genre', 'Count']]

print("Top ten most popular genres for each split")
# Display the results
print(combined_genres_df)
print("Table for Top ten most popular genres for each split")
# Print the DataFrame using tabulate with grid format
print(tabulate(combined_genres_df, headers='keys', tablefmt='grid'))

# Stop the Spark session
spark.stop()
