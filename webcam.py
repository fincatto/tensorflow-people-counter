#import os
import cv2
#import tarfile
#import tensorflow as tf
#import six.moves.urllib as urllib

# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#MODEL_FILE = 'ssd_inception_v2_coco_2017_11_17.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#DOWNLOAD_DIR = 'downloads/'
#DOWNLOAD_FILE = DOWNLOAD_DIR + MODEL_FILE

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = DOWNLOAD_DIR + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Number of classes to detect
#NUM_CLASSES = 90

# Download Model
#if not os.path.exists(PATH_TO_CKPT):
#    print ('Criando diretorio do model:', DOWNLOAD_DIR)
#    if not os.path.exists(DOWNLOAD_DIR):
#        os.makedirs(DOWNLOAD_DIR)
#
#    print ('Iniciando download do model:',  DOWNLOAD_FILE)
#    opener = urllib.request.URLopener()
#    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, DOWNLOAD_FILE)
#    tar_file = tarfile.open(DOWNLOAD_FILE)
#    for file in tar_file.getmembers():
#        file_name = os.path.basename(file.name)
#        if 'frozen_inference_graph.pb' in file_name:
#            print ('Estraindo arquivo em', PATH_TO_CKPT)
#            tar_file.extract(file, PATH_TO_CKPT)

# Define the video stream
#print ('Iniciando captura de video...')
#cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
#print(cap)


def show_webcam():
    print ('Iniciando captura de video...')
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read() #captura a imagem da webcam
        img = cv2.flip(img, 1) #gira imagem para parecer um espelho
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transforma a imagem em escala de cinza
        
        #para cada foto capturada da webcam, exibo em uma tela para analise
        cv2.imshow('CV2 Video Capture', img)
        
        # esc para sair
        if cv2.waitKey(1) == 27: 
            print ('Gravando ultima imagem capturada em /tmp/img.png...')
            cv2.imwrite('/tmp/img.png',img) #grava a imagem capturada
            break #sai do loop  
    
    # libera os recursos
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_webcam()