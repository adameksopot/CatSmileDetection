import logging
import os
from sklearn.metrics import accuracy_score
import seaborn as sns
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D,MaxPooling2D, Dropout,Flatten,Dense,Activation, BatchNormalization
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
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

batch_size = 28
img_height = 28
img_width = 28

# przygotowanie treningowych i walidacyjnych:
train_ds = tf.keras.utils.image_dataset_from_directory(
  'C:\\Users\\Zuzanna.Deska\\PycharmProjects\\studia\\CatSmileDetection\\data_for_model',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'C:\\Users\\Zuzanna.Deska\\PycharmProjects\\studia\\CatSmileDetection\\data_for_model',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True)


# przygotowanie testowych:
test_ds = tf.keras.utils.image_dataset_from_directory(
  'C:\\Users\\Zuzanna.Deska\\PycharmProjects\\studia\\CatSmileDetection\\tests',
  validation_split=0.01,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=99)



class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")


num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 2, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(5000, activation='relu'),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
model.save('saved_model/my_model')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

img = tf.keras.utils.load_img(
    "C:\\Users\\Zuzanna.Deska\\PycharmProjects\\studia\\CatSmileDetection\\test1.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))

for images, labels in test_ds.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
# evaluate the model
y_pred = model.predict(numpy_images, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

loss, accuracy = model.evaluate(test_ds, verbose=0)

#loss, accuracy = model.evaluate(test_ds, verbose=2)
print("loss: {}, acc: {}".format(loss, accuracy))

x = accuracy_score(numpy_labels, y_pred_classes)
print(classification_report(numpy_labels, y_pred_classes, target_names=class_names))

matrix = confusion_matrix(numpy_labels, y_pred_classes)
labels1 = ['neutral', 'smiling']
plt.subplots(figsize=(10, 10))
sns.heatmap(matrix, annot=True, annot_kws={"size": 10}, fmt='.1f', xticklabels=class_names, yticklabels=class_names)
logging.info('done')