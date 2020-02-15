import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

def config_logging():
    formatter = log.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] - %(message)s")
    rootLogger = log.getLogger()

    fileHandler = log.FileHandler("/tmp/wc2.log")
    fileHandler.setFormatter(formatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = log.StreamHandler()
    consoleHandler.setFormatter(formatter)
    rootLogger.addHandler(consoleHandler)

def start_detection():
    log.debug("Iniciando sistema de identificacao de faces...")
    try:
        cap = cv2.VideoCapture(0)
        anterior = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if anterior != len(faces):
                anterior = len(faces)
                #log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
                print("Faces: "+str(len(faces))+" em "+str(dt.datetime.now()))

            # Display the resulting frame
            cv2.imshow('Video', frame)

            #sai da tela com esc
            if cv2.waitKey(1) == 27: 
                break
    except Exception as e:
        log.error('Erro ao rodar sistema: '+ str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    config_logging()
    start_detection()
    