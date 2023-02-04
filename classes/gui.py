import os
from tkinter import Tk, Label, Button, Canvas
from tkinter import filedialog
from tensorflow import keras
import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk
import cv2.cv2 as cv2
import tensorflow as tf
from keras import backend as K


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def f1_score(y_true, y_pred):
    precision1 = precision(y_true, y_pred)
    recall1= recall(y_true, y_pred)
    return 2 * ((precision1 * recall1) / (precision1 + recall1 + K.epsilon()))


class MyFirstGUI(Tk):
    def __init__(self):
        super(MyFirstGUI, self).__init__()
        self.title("Python Tkinter Dialog Widget")
        self.minsize(420, 320)

        # self.labelFrame = ttk.LabelFrame(self, text="Wybierz plik")
        # self.labelFrame.grid(column=0, row=0, padx=20, pady=20)
        # self.labelFrame1 = ttk.LabelFrame(self, text="Predykcja")
        # self.labelFrame1.grid(column=0, row=1, padx=20, pady=50)
        l1 = Label(self, text="Select image:")
        l2 = Label(self, text="Prediction:")
        l1.grid(row=1, column=0)
        l2.grid(row=1, column=2)
        self.canvas1 = Canvas(bg="white", width=420, height=290)
        self.canvas1.grid(row = 0, column = 0, columnspan = 4)
        self.button = ttk.Button(self, text="Browse a file and get prediction", command=self.browse_and_cut)
        self.button.grid(column=1, row=1)

    def predict(self):
        model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'saved_model/my_model'))
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    def browse_and_cut(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.label = ttk.Label(self, text="")
        self.label.grid(column=0, row=1)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")
        img = cv2.imread(self.filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5,minSize=[80, 80])
        for x, y, w, h in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        crop_img = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
        resized = cv2.resize(crop_img, (28,28))
        model = keras.models.load_model(os.path.join(os.getcwd(), 'model/saved_model/my_model'), custom_objects={'precision':precision, 'recall': recall, 'f1_score':f1_score})
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        img_array = tf.keras.utils.img_to_array(resized)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        photo = ImageTk.PhotoImage(image=Image.fromarray(img[...,::-1]))

        self.label2 = Label(image=photo)
        self.label2.image = photo
        self.label2.grid(column=0, row=0, columnspan=4)
        class_names = ['neutral', 'smiling']
        self.label1 = ttk.Label(self, text=class_names[np.argmax(score)])
        self.label1.grid(column=3, row=1)


    def fileDialog(self):
        pass
