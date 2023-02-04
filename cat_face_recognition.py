import os

import cv2.cv2 as cv2


#images1 = os.listdir('out_smile')

# img = [i for i in images1 if i in images]
#
# for i in img:
#     os.remove('out/{}'.format(i))
# print('d')
images = os.listdir('out')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")
for i, image in enumerate(images):
    img = cv2.imread(os.path.join("out", image))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=[80, 80])

    if len(faces) != 0:
        crop_img = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
        filename = images[i]
        cv2.imwrite('out_smiling_cut/{}'.format(filename), crop_img)

#
# print('done')