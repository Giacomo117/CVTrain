## Folder structure
The YawnDD folder contains everything needed to deal with [YasDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) dataset basic training.

### Training
Training contains 3 relevant scripts and one txt. The 3 scripts are:
- DatasetDownloader.py, which downloads the YawDD from IEEE website (skipping the authentication requirements) and extracts it
- DatasetPreparation.py, which extracts the frames from the side videos, dividing them into different folders based on if the frame contains a yawn or not
- ModelTraining.py, which is the actual training script
- finalenames.txt is used to keep track of which extracted frames do actually contain a yawn. Basically, it contains the labels used in DatasetPreparation.py

```bash
cd ./YawnDD
python ./Training/DatasetDownloader.py
python ./Training/DatasetPreparation.py 
python ./Training/ModelTraining-grayscale.py
```

### Visualization
In this folder, we contain the only relevant script, which is visualization.py

```bash
cd ./YawnDD
python ./Visualization/visualization.py
```