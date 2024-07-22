### Steps
1. Download mrlEyes
2. DataPreparation.py --> Prepares the dataset in a supervised manner (splitting into two folders)
3. mobilelowparams.py (in Final Model) --> Trains the model and saves it as a .keras file
4. convert.py --> Converts the .keras file from the server to the required .h5 format
5. launch.py --> Main script that, given the models, uses everything in real-time on the computer's camera

Change the directory from which it takes the images
Dataset used: mrlEyes-2018 (80000 images) with data augmentation 1:5)

pip install -r requirements.txt
