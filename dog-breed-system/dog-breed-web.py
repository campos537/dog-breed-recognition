from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.session import set_env, info as session_info

import cv2
from utils.dog_classifier import DogClassifier
import numpy as np
import base64
import json
import os

def classificacao(dog_class):
    with use_scope('A'):
        img = file_upload("Escolha a imagem :", accept="image/*", help_text='Apos escolher a imagem clique no botao "Submit"')
        if img is None:
            put_markdown("`img = %r`" % img)
        else:
            clear('A')
            image = bytes_to_np(img['content'])
            image = cv2.resize(image, (512, 512))
            dog_class.classify(image)
            put_image(cv2.imencode('.png', image)[1].tobytes(),height = '512px', width='512px')

def bytes_to_np(byte_img):
    img = cv2.imdecode(np.frombuffer(byte_img, np.uint8), -1)
    final_img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    return final_img


def main():
    put_markdown("""# Dog Breed Web System

    Essa aplicacao testa de forma simples as solucoes feitas para o reconhecimento de racas de cachorro    

    ### **Classificacao das racas**
    """, lstrip=True)
    
    dog_class = DogClassifier(0.6)

    while True:
        classificacao(dog_class)

if __name__ == '__main__':
    start_server(main, debug=True, port=8080, cdn=False)