from tkinter import *
from tkinter.font import Font
from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import Tk, Canvas, Frame, BOTH, NW
import json

from threading import Thread
import threading

import cv2
from utils.dog_classifier import DogClassifier
from utils.dog_descriptor import DogDescriptor
import os


class MainWindow:

    def browse_files(self):
        file_name = filedialog.askopenfilename(initialdir="./",
                                               title="Select a File",
                                               filetypes=(("Image Files",
                                                           "*.png ; *.jpg ; *.jpeg"),
                                                          ("All Files", "*.*")))
        self.path_text.delete("1.0", END)
        self.path_text.insert("1.0", file_name)

    def browse_folder(self):
        folder_name = filedialog.askdirectory(initialdir="./",
                                              title="Select a Folder")
        self.path_folder_text.delete("1.0", END)
        self.path_folder_text.insert("1.0", folder_name)

    def browse_files_enroll(self):
        file_name = filedialog.askopenfilename(initialdir="./",
                                               title="Select a File",
                                               filetypes=(("Image Files",
                                                           "*.png ; *.jpg ; *.jpeg"),
                                                          ("All Files", "*.*")))
        self.path_text_enroll.delete("1.0", END)
        self.path_text_enroll.insert("1.0", file_name)

    def add_method(self):
        folder = self.path_folder_text.get("1.0", END).replace("\n", "")
        race_name = self.path_raca_text.get("1.0", END).replace("\n", "")

        if os.path.isdir(folder) and len(race_name) > 0 and threading.activeCount() < 3:

            carregando = Image.open("assets/carregando.jpg")
            carregando = carregando.resize((512, 512), Image.ANTIALIAS)
            carregando_tk = ImageTk.PhotoImage(carregando)
            self.video_image_label.configure(image=carregando_tk)
            self.video_image_label.image = carregando_tk
            self.video_image_label.place_configure(relx=0.40, rely=0.1)

            DogDescriptor.add_new_dog(folder, race_name)

            done = Image.open("assets/pata.jpg")
            done = done.resize((512, 512), Image.ANTIALIAS)
            done_tk = ImageTk.PhotoImage(done)
            self.video_image_label.configure(image=done_tk)
            self.video_image_label.image = done_tk
            self.video_image_label.place_configure(relx=0.40, rely=0.1)

    def clean(self):
        DogDescriptor.descriptors = {}

    def test_enroll(self):
        test_enroll_img = self.path_text_enroll.get(
            "1.0", END).replace("\n", "")
        if os.path.isfile(test_enroll_img) and threading.activeCount() < 3:
            img = cv2.imread(test_enroll_img)
            img = cv2.resize(img, (512, 512))

            DogDescriptor.recognize(img)

            im = Image.fromarray(img)
            b, g, r = im.split()
            im = Image.merge("RGB", (r, g, b))
            video_image = ImageTk.PhotoImage(image=im)
            self.video_image_label.configure(image=video_image)
            self.video_image_label.image = video_image
            self.video_image_label.place_configure(relx=0.40, rely=0.1)

    def process_dog(self):
        test_class_img = self.path_text.get("1.0", END).replace("\n", "")
        if os.path.isfile(test_class_img) and threading.activeCount() < 3:
            dog_class = DogClassifier(0.6)
            img = cv2.imread(test_class_img)
            img = cv2.resize(img, (512, 512))
            dog_class.classify(img)

            im = Image.fromarray(img)
            b, g, r = im.split()
            im = Image.merge("RGB", (r, g, b))
            video_image = ImageTk.PhotoImage(image=im)
            self.video_image_label.configure(image=video_image)
            self.video_image_label.image = video_image
            self.video_image_label.place_configure(relx=0.40, rely=0.1)

    def start_class(self):
        self.thread = Thread(
            target=MainWindow.process_dog, kwargs={"self": self})
        self.thread.start()

    def start_enroll(self):
        self.thread = Thread(
            target=MainWindow.test_enroll, kwargs={"self": self})
        self.thread.start()

    def add(self):
        self.thread = Thread(
            target=MainWindow.add_method, kwargs={"self": self})
        self.thread.start()

    def classification_part(self, win, pata_tk):
        # Essa parte do codigo pega a imagem para fazer a inferencia no modelo de classificacao
        # e dar como output a imagem com as top5 classes possiveis.
        self.class_label = Label(
            win, text="Teste Imagem (Parte 1 Classificacao)", font=(None, 13))
        self.class_label.place_configure(relx=.02, rely=.128)
        self.button_explore = Button(win,
                                     text="...",
                                     command=self.browse_files)
        self.button_explore.place(
            relx=.29, rely=.160, width=30, height=30, bordermode="inside")
        self.path_text = Text(win, font="Times32", width=25, height=1.4)
        self.path_text.place(relx=.15, rely=.176, anchor="center")

        self.button_start_class = Button(win,
                                         text="Rodar", image=pata_tk,
                                         command=self.start_class)
        self.button_start_class.image = pata_tk
        self.button_start_class.place(
            relx=.15, rely=.190, width=41, height=41, bordermode="inside")

    def enroll_part(self, win, pata_tk):
        #  Essa parte do codigo faz a adicao de novas racas de cachorro, passe a pasta que quer
        # adicionar e a label que o cachorro vai possuir,
        self.enroll_label = Label(
            win, text="Adicione Cachorros (Parte 2 Enroll)", font=(None, 13))
        self.enroll_label.place_configure(relx=.02, rely=.240)
        self.button_folder = Button(win,
                                    text="...",
                                    command=self.browse_folder)
        self.button_folder.place(
            relx=.29, rely=.272, width=30, height=30, bordermode="inside")
        self.path_folder_text = Text(win, font="Times32", width=25, height=1.4)
        self.path_folder_text.place(relx=.15, rely=.288, anchor="center")

        self.label_label = Label(win, text="Nome da Raca: ")
        self.label_label.place_configure(relx=.03, rely=.320)

        self.path_raca_text = Text(win, font="Times32", width=15, height=1.4)
        self.path_raca_text.place(relx=.23, rely=.330, anchor="center")
        self.add_race = Button(win,
                               text="Adicionar",
                               command=self.add)
        self.add_race.place(
            relx=.06, rely=.350, width=100, height=30, bordermode="inside")

        self.clean_races = Button(win,
                                  text="Limpar Enroll",
                                  command=self.clean)
        self.clean_races.place(
            relx=.18, rely=.350, width=100, height=30, bordermode="inside")

        # Essa parte continua a implementacao do Enroll, porem serve para escolher a imagem que sera testada
        self.enroll_label = Label(
            win, text="Teste Imagem Enroll (Parte 2 Enroll)", font=(None, 13))
        self.enroll_label.place_configure(relx=.02, rely=.400)
        self.button_explore_enroll = Button(win,
                                            text="...",
                                            command=self.browse_files_enroll)
        self.button_explore_enroll.place(
            relx=.29, rely=.432, width=30, height=30, bordermode="inside")
        self.path_text_enroll = Text(win, font="Times32", width=25, height=1.4)
        self.path_text_enroll.place(relx=.15, rely=.448, anchor="center")

        self.button_start_enroll = Button(win,
                                          text="Rodar", image=pata_tk,
                                          command=self.start_enroll)
        self.button_start_enroll.image = pata_tk
        self.button_start_enroll.place(
            relx=.15, rely=.462, width=41, height=41, bordermode="inside")

    def __init__(self, win):
        font_style = Font(root=win.master, size=12,
                          weight="bold", family="Arial")

        dog_caramelo = Image.open("assets/caramelo.jpg")
        dog_caramelo = dog_caramelo.resize((960, 960), Image.ANTIALIAS)
        dog_caramelo_tk = ImageTk.PhotoImage(dog_caramelo)
        self.caramelo_image = Label(win, image=dog_caramelo_tk)
        self.caramelo_image.image = dog_caramelo_tk
        self.caramelo_image.place_configure(relx=0.0, rely=0.0)

        self.video_image_label = Label(win)

        pata = Image.open("assets/pata.jpg")
        pata = pata.resize((41, 41), Image.ANTIALIAS)
        pata_tk = ImageTk.PhotoImage(pata)

        self.classification_part(win, pata_tk)
        self.enroll_part(win, pata_tk)
