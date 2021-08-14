import cv2
import os
from shutil import copy

dataset_path = "/media/disk1/Repositorios/UnicoID/dogs-project/dataset/train"
output_dataset =  "/media/disk1/Repositorios/UnicoID/dogs-project/dataset/train_cleaned"
output_test_dataset =  "/media/disk1/Repositorios/UnicoID/dogs-project/dataset/test_cleaned"
count = 0
percent = 0.1
for dog_class in os.listdir(dataset_path):
    image_list = os.listdir(dataset_path+"/"+dog_class)
    total_test = int(len(image_list) * percent)
    count_class = 0
    for image in image_list:
        img = cv2.imread(dataset_path+"/"+dog_class+"/"+image, 0)
        if cv2.countNonZero(img) == 0:
            count+=1
        else:
            dog_normal = dog_class.split("-")[1]
            if not os.path.isdir(output_dataset+"/"+dog_normal+"/"):
                os.mkdir(output_dataset+"/"+dog_normal+"/")
            if not os.path.isdir(output_test_dataset+"/"+dog_normal+"/"):
                os.mkdir(output_test_dataset+"/"+dog_normal+"/")
            
            if count_class < total_test:        
                copy(dataset_path+"/"+dog_class+"/"+image, output_test_dataset+"/"+dog_normal+"/"+image)
            else:
                copy(dataset_path+"/"+dog_class+"/"+image, output_dataset+"/"+dog_normal+"/"+image)
            count_class +=1
print("Imagens pretas: ", count)