#!/bin/bash
#SBATCH --job-name=Q1_script # Replace JOB_NAME with a name you like
#SBATCH --time=00:30:00  # Change this to a longer time if you need more time
#SBATCH --nodes=2  # Specify a number of nodes
#SBATCH --mem=5G  # Request 5 gigabytes of real memory (mem)
#SBATCH --output=./Output/Q1_output.txt  # This is where your output and errors are logged
#SBATCH --mail-user=pksasikumar1@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

spark-submit /users/acp23pks/com6012/ScalableML/Code/Q1_code.py  # .. is a relative path, meaning one level up


