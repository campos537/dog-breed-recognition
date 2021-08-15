import cv2
from predictor.predictor import Predictor 
from utils.config import Config
import pandas as pd
import numpy as np
import json


class DogClassifier:

    def __init__(self,threshold_unknown):
        config = Config("config_class.json")
        self.dog_model = Predictor(config)
        self.threshold_unknown = threshold_unknown

    def classify(self, image):
        predictions = self.dog_model.classify_topn(image, 3)
        return self.draw_dog_results(image, predictions)

    def check_unknown(self,result):
        return not (result[0][1] > 0.6)

    def draw_dog_results(self, image, result):
        if not self.check_unknown(result):
            cv2.putText(image, result[0][0] + " " + str(round(result[0][1]*100))+"%", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)
            cv2.putText(image, result[1][0] + " " + str(round(result[1][1]*100))+"%", (0, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)
            cv2.putText(image, result[2][0] + " " + str(round(result[2][1]*100))+"%", (0, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)
        else:
            cv2.putText(image, "Raca Desconhecida", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
        return image