import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, dayofmonth, hour, max, regexp_extract, split, size
from pyspark.sql.functions import to_timestamp
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Q1") \
    .config("spark.local.dir","/mnt/parscratch/users/acp23pks") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Read and cache the log file
logFile = spark.read.text("/users/acp23pks/com6012/ScalableML/Data/NASA_access_log_Jul95.gz").cache()

# Define a regex pattern for host and country code extraction
host_pattern = r'^(\S+)'
country_pattern = r'\.([a-z]{2})$'

print(f"Total lines in the original file: {logFile.count()}")

# Split the log data into columns
data = logFile.withColumn('host', regexp_extract('value', host_pattern, 1)) \
                .withColumn('timestamp', regexp_extract('value', r'.* - - \[(.*)\].*', 1)) \
                .withColumn('request', regexp_extract('value', r'.*\"(.*)\".*', 1)) \
                .withColumn('HTTP reply code', split('value', ' ').getItem(size(split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', split('value', ' ').getItem(size(split('value', ' ')) - 1).cast("int")) \
                .drop("value")

# Drop rows with missing values and cache the result for subsequent actions
data = data.na.drop()
data.cache()

# Print a sample of the data
data.show(20, truncate=False)

# Define a function to count requests by country code
def count_requests_by_country(country_code):
    return data.filter(data['host'].endswith(country_code)).count()

request_count_germany = count_requests_by_country('.de')
request_count_canada = count_requests_by_country('.ca')
request_count_singapore = count_requests_by_country('.sg')

data.filter(data['host'].rlike('.*\.de$')).show(5, False)
data.filter(data['host'].rlike('.*\.ca$')).show(5, False)
data.filter(data['host'].rlike('.*\.sg$')).show(5, False)

print("==================== Task A ====================")


# Print the total number of requests for each country
print(f"Total requests from Germany: {request_count_germany}")
print(f"Total requests from Canada: {request_count_canada}")
print(f"Total requests from Singapore: {request_count_singapore}")

# Plot the bar chart for total number of requests by country
countries = ['Germany', 'Canada', 'Singapore']
counts = [request_count_germany, request_count_canada, request_count_singapore]
plt.bar(countries, counts, color=['red', 'yellow', 'purple'])

#add title and label
plt.title('Number of Requests by Country')
plt.xlabel('Country')
plt.ylabel('Number of Requests')

plt.savefig('/users/acp23pks/com6012/ScalableML/Output/Q1_figA.jpg', dpi=200, bbox_inches="tight")
plt.close()

print("==================== Task B ====================")

# Filter for unique hosts
uniqueHostsGermany = data.filter(F.col('host').rlike('.*\.de$')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']
uniqueHostsCanada = data.filter(F.col('host').rlike('.*\.ca$')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']
uniqueHostsSingapore = data.filter(F.col('host').rlike('.*\.sg$')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']

print(f"Germany has {uniqueHostsGermany} unique hosts.")
print(f"Canada has {uniqueHostsCanada} unique hosts.")
print(f"Singapore has {uniqueHostsSingapore} unique hosts.")


# Fetch and print top host data for each country
topHostsGermany = data.filter(data.host.endswith('.de')).groupBy('host').count().orderBy(desc('count')).limit(9).collect()
topHostsCanada = data.filter(data.host.endswith('.ca')).groupBy('host').count().orderBy(desc('count')).limit(9).collect()
topHostsSingapore = data.filter(data.host.endswith('.sg')).groupBy('host').count().orderBy(desc('count')).limit(9).collect()

# Define a function to display unique and most active hosts
def display_country_data(country_name, unique_hosts, top_hosts):
    print(f"\n{country_name} Analysis")
    print("-" * 40)
    print(f"Unique Hosts in {country_name}: {unique_hosts}\n")
    
    print(f"Top 9 Most Active Hosts in {country_name}:")
    top_hosts_df = pd.DataFrame(top_hosts, columns=['Host', 'Count'])
    print(top_hosts_df.to_string(index=False))

# Function to retrieve and display host data
def get_and_display_host_data(data, country_code, country_name):
    unique_hosts = data.filter(F.col('host').rlike(f'.*\\.{country_code}$')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']
    
    top_hosts = data.filter(data.host.endswith(f'.{country_code}'))\
                    .groupBy('host')\
                    .count()\
                    .orderBy(desc('count'))\
                    .limit(9)\
                    .collect()
    top_hosts_list = [(row['host'], row['count']) for row in top_hosts]
    
    display_country_data(country_name, unique_hosts, top_hosts_list)

# Call the function for each country
get_and_display_host_data(data, 'de', 'Germany')
get_and_display_host_data(data, 'ca', 'Canada')
get_and_display_host_data(data, 'sg', 'Singapore')



print("==================== Task C ====================")
# Function to count requests by country code
def calculate_percentages(country_code, country_name):
    top_hosts_df = data.filter(data.host.endswith(country_code)).groupBy('host').count().orderBy(desc('count'))
    top_hosts_counts = top_hosts_df.limit(9)
    top_hosts_list = [(row['host'], row['count']) for row in top_hosts_counts.collect()]
    
    total_requests = count_requests_by_country(country_code)
    total_top_hosts_requests = sum([count for _, count in top_hosts_list])
    remaining_requests = total_requests - total_top_hosts_requests
    
    top_hosts_list.append(('Rest', remaining_requests))
    hosts, counts = zip(*top_hosts_list)
    
    percentages = [count / total_requests * 100 for count in counts]
    
    plt.figure(figsize=(12, 8))  
    bars = plt.bar(hosts, percentages, color='red', width=0.6)  
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10, color='black')
    
    plt.title(f'Percentage of Requests by Host for {country_name}', fontsize=14)
    plt.xlabel('Hosts', fontsize=12)
    plt.ylabel('Percentage of Total Requests', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  
    
    # Save the chart 
    plt.savefig(f'/users/acp23pks/com6012/ScalableML/Output/Q1_figC_{country_name}.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
print("==================== Task D ====================")

# Execute the function for each country
calculate_percentages('.de', 'Germany')
calculate_percentages('.ca', 'Canada')
calculate_percentages('.sg', 'Singapore')

# Define the countries 
countries = ['Germany', 'Canada', 'Singapore']

# Define a function to extract the day and hour from the timestamp
def extract_day_hour(timestamp):
    day = timestamp.split(':')[0][0:2]
    hour = timestamp.split(':')[1]
    return (int(hour), int(day))

# Define a function to plot the heatmap
def plot_heatmap(data, title):
    # Create a 2D array with the number of visits for each hour of each day
    heatmap_data = np.zeros((24, 31))
    for row in data:
        day, hour, count = row
        heatmap_data[hour, day-1] = count

    # Create the plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(title)
    ax.set_ylabel('Hour')
    ax.set_xlabel('Day')
    ax.set_yticks(np.arange(0, 24))
    ax.set_xticks(np.arange(1, 32))
    ax.set_yticklabels(np.arange(0, 24))
    ax.set_xticklabels(np.arange(1, 32))
    
    heatmap = ax.pcolor(heatmap_data, cmap=plt.cm.Reds, edgecolors='k', linewidths=0.5)
    colorbar = plt.colorbar(heatmap)
    colorbar.set_label('Number of Visits')
    plt.savefig('/users/acp23pks/com6012/ScalableML/Output/Q1_figD_{}.jpg'.format(title), dpi=200, bbox_inches="tight")
    plt.close()
    
# Filter the data for the top host from each country
topHostsGermany = data.filter(data.host.like('%.de')).groupBy('host').count().orderBy(desc('count')).limit(1).collect()[0].host
topHostsCanada = data.filter(data.host.like('%.ca')).groupBy('host').count().orderBy(desc('count')).limit(1).collect()[0].host
topHostsSingapore = data.filter(data.host.like('%.sg')).groupBy('host').count().orderBy(desc('count')).limit(1).collect()[0].host


# Extract the day and hour of the visit from the timestamp
data_G = data.select('timestamp').filter((data.host == topHostsGermany))
data_C = data.select('timestamp').filter((data.host == topHostsCanada))
data_S = data.select('timestamp').filter((data.host == topHostsSingapore))
data_G = data_G.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])
data_C = data_C.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])
data_S = data_S.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])

# Group the data by day and hour and count the number of visits
data_G = data_G.groupBy(['day', 'hour']).count().orderBy(['day', 'hour'])
data_C = data_C.groupBy(['day', 'hour']).count().orderBy(['day', 'hour'])
data_S = data_S.groupBy(['day', 'hour']).count().orderBy(['day', 'hour'])
data_C.orderBy(desc('day'), desc('hour'))
data_C.orderBy(('day'), ('hour'))
data_G.orderBy(desc('day'), desc('hour'))
data_G.orderBy(('day'), ('hour'))
data_S.orderBy(('day'), ('hour'))
data_S.orderBy(desc('day'), desc('hour'))

# Plot the heatmap for each country
data_germany = data_G.filter(data_G.day <= 31)
plot_heatmap(data_germany.collect(), 'Germany')
data_canada = data_C.filter(data_C.day <= 31)
plot_heatmap(data_canada.collect(), 'Canada')
data_singapore = data_S.filter(data_S.day <= 31)
plot_heatmap(data_singapore.collect(), 'Singapore')

spark.stop()






