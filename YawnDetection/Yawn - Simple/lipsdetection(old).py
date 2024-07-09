import cv2
import requests

def detect_lips():
    # Carica il classificatore Haar cascade per le labbra
    lips_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Accedi alla fotocamera
    cap = cv2.VideoCapture(0)

    while True:
        # Leggi l'immagine dalla fotocamera
        ret, img = cap.read()

        # Converti l'immagine in scala di grigi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Rileva le labbra
        lips = lips_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)

        # Disegna un rettangolo attorno alle labbra rilevate e mostra l'altezza del rettangolo
        for (x, y, w, h) in lips:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, "Height: " + str(h), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(img, "Lips detected: " + str(len(lips)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostra l'immagine
        cv2.imshow('img', img)

        # Esci se 'q' è premuto
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la fotocamera e distrugge tutte le finestre
    cap.release()
    cv2.destroyAllWindows()


def download_haarcascade_file():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
    response = requests.get(url)

    # Assicurati che la richiesta sia stata eseguita correttamente
    if response.status_code == 200:
        # Scrivi il contenuto in un file
        with open("haarcascade_smile.xml", "wb") as file:
            file.write(response.content)
    else:
        print("Errore durante il download del file: status code", response.status_code)

def detect_lips_and_keypoints(source=0):
    # Carica il classificatore Haar cascade per le labbra
    lips_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Crea l'oggetto SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Accedi alla sorgente video
    cap = cv2.VideoCapture(source)

    while True:
        # Leggi l'immagine dalla sorgente video
        ret, img = cap.read()

        # Se non c'è immagine, interrompi il ciclo
        if img is None:
            break

        # Converti l'immagine in scala di grigi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Rileva le labbra
        lips = lips_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=100)

        # Disegna un rettangolo attorno alle labbra rilevate e mostra l'altezza del rettangolo
        for (x, y, w, h) in lips:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, "Height: " + str(h), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Rileva i keypoints solo nelle labbra
            roi = gray[y:y+h, x:x+w]
            keypoints = sift.detect(roi, None)

            # Converti le coordinate dei keypoints per adattarle all'immagine originale
            keypoints = [cv2.KeyPoint(kp.pt[0]+x, kp.pt[1]+y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

            # Disegna i keypoints sull'immagine
            cv2.drawKeypoints(img, keypoints, img)

        cv2.putText(img, "Lips detected: " + str(len(lips)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostra l'immagine
        cv2.imshow('img', img)

        # Esci se 'q' è premuto
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la sorgente video e distrugge tutte le finestre
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    download_haarcascade_file()
    detect_lips_and_keypoints("C:\\Users\\stopp\\OneDrive - E38\Desktop\\CVTrain\\YawnDetection\\YawnDD\\yawdd\\YawDD dataset\\Mirror\\Male_mirror Avi Videos\\2-MaleGlasses-Yawning.avi")