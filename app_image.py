import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import cv2
import sqlite3 as sql

model = model_from_json(open('i.json', 'r').read())
model.load_weights('fer.h5')
cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
to1 = input("write path to the folder where app is located: ")
to = input("write path to your image: ")
s = sql.connect(to1 + "/image_out/imout.db")
cur = s.cursor()
c = cv2.imread(to)
g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
o = ""
face = cas.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5)
for (x, y, z, w) in face:
    cv2.rectangle(c, (x, y), (x + z, y + w), (254, 0, 0), thickness=5)
    i = g[y: y + z, x: x + w]
    f = cv2.resize(i, (48, 48))
    im = image.img_to_array(f)
    dir = np.expand_dims(im, axis=0)
    dir /= 255
    pred = model.predict(dir)
    u = np.argmax(pred[0])
    k = ('angry', 'disgust', 'fear', 'laughter', 'sad', 'surprise', 'neutral')
    o = k[u]
    cv2.putText(c, o, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
p = cv2.resize(c, (1000, 700))
cv2.imwrite(to1 + "/image_out/im.jpg", c)
numr = cur.execute(""" SELECT num from img """).fetchall()
nnn = len(numr) + 1
nnn = nnn - 1
cur.execute(""" INSERT INTO img(num, emotion) VALUES(?, ?)""", (nnn, o))
s.commit()
s.close()
#cv2.imshow('jeb', p)
cv2.destroyAllWindows()