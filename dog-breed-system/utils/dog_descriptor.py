import cv2
from predictor.predictor import Predictor
from utils.config import Config
import pandas as pd
from scipy import spatial
import os
import numpy as np
import json


class DogDescriptor:
    threshold_unknown = 0.65
    descriptors = {}
    config = Config("config_desc.json")
    desc_dog_model = Predictor(config)

    @classmethod
    def add_new_dog(cls, folder, race_name):
        # t1 = cv2.TickMeter()
        if cls.descriptors.get(race_name) is None:
            cls.descriptors[race_name] = [0, []]
        for image in os.listdir(folder):
            # t1.start()
            img = cv2.imread(folder+"/"+image)
            if not img.size == 0:
                descriptor = cls.desc_dog_model.predict(img)
                if cls.descriptors[race_name][0] == 0:
                    cls.descriptors[race_name][1] = descriptor
                    cls.descriptors[race_name][0] += 1
                else:
                    cls.descriptors[race_name][0] += 1
                    numerator = cls.descriptors[race_name][1] * \
                        (cls.descriptors[race_name][0] - 1)
                    cls.descriptors[race_name][1] = (
                        descriptor+cls.descriptors[race_name][1])/cls.descriptors[race_name][0]
            # t1.stop()
            # print(t1.getTimeMilli())
            # t1.reset()

    @classmethod
    def recognize(cls, image):
        if len(cls.descriptors) < 1:
            cv2.putText(image, "Raca Desconhecida", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
        else:
            descriptor = cls.desc_dog_model.predict(image)

            match = ""
            max_sim = -99999
            for race in cls.descriptors:
                similiarity = 1 - \
                    spatial.distance.cosine(
                        cls.descriptors[race][1], descriptor)
                if similiarity > cls.threshold_unknown and similiarity > max_sim:
                    match = race

            if len(match) > 0:
                cv2.putText(image, match, (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)
            else:
                cv2.putText(image, "Raca Desconhecida", (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
