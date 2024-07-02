#!/bin/bash
#SBATCH --job-name=training_prompt
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G

#SBATCH --account=cvcs2024
#SBATCH --output="logs/promptingmobile.log"
#SBATCH --error="logs/promptingmobile.log"
cd /homes/slugli/CVTrain/YawnDetection/YawnDD
python /homes/slugli/CVTrain/YawnDetection/YawnDD/Training/ModelTraining.py
