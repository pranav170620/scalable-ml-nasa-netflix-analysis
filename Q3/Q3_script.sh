#!/bin/bash
#SBATCH --job-name=Q3_script # Replace JOB_NAME with a name you like
#SBATCH --time=06:00:00  # Change this to a longer time if you need more time
#SBATCH --nodes=10  # Specify a number of nodes
#SBATCH --mem=20G  # Request 20 gigabytes of real memory (mem)
#SBATCH --output=./Output/Q3_output.txt # This is where your output and errors are logged
#SBATCH --mail-user=pksasikumar1@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

spark-submit --driver-memory 20g --executor-memory 20g /users/acp23pks/com6012/ScalableML/Code/Q3_code.py  

