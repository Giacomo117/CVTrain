
Steps
1) Scarica mrlEyes
2) DataPreparation.py --> Prepara il dataset in modo supervisionato (divisione in due cartelle)
3) DataTrainServer.py --> versione da usare sul server unimore per trainare la rete
  DataTrainDrive se lo si usa su colab (prima carica il dataset su drive)
  HMCNN --> CNN fatta a mano che però va malino
  ServerV2 --> in queue for training

4) convert.py --> converte il .keras del server al .h5 che serve 
5) launch.py --> main che, dati i modelli, usa tutto real-time sulla telecamera del computer

Cambiare directory da cui prende le immagini
Eye Detection
Dataset usato: mrlEyes-2018 (80000 immagini) con data augmentation 1:5)
pip install -r requirements.txt
