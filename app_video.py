import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import cv2
import sqlite3 as sql
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp
import ffmpeg

model = model_from_json(open('i.json', 'r').read())
model.load_weights('fer.h5')

cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
to1 = input("write path to the folder where app is located: ")
to = input("write path to your video: ")
size_x = input("write width of your video: ")
size_y = input("write height of your video: ")
s = sql.connect(to1 + "/video_out/vout.db")
cur = s.cursor()
cam = cv2.VideoCapture(to)
audio = mp.AudioFileClip(to)
cam.set(3, 640)
cam.set(4, 480)
sr = cv2.VideoWriter_fourcc(*"XVID")
p = 1
d = 0
j = 0
fr = []
tr = []
thr = ['two']
g1 = 0
y = 0
out = cv2.VideoWriter(to1 + '/video_out/output_vid.avi', sr, 20.0, (int(size_x), int(size_y)))
u = True
try:
    while u:
        k, c = cam.read()
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
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
            k = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutralness')
            o = k[u]
            thr.append(o)
            cv2.putText(c, o, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if (x, y, z, w):
                # cv2.imwrite(to1 + '/video_out/cap' + str(p) + '.jpg', c)
                numr = cur.execute(""" SELECT num from frame """).fetchall()
                nnn = len(numr) + 1
                nnn = nnn - 1
                pos = cam.get(cv2.CAP_PROP_POS_MSEC)
                sec = pos / 1000
                fr.append(sec)
            r = j + 1
            if y > 0:
                print(thr)
                if thr[d] is thr[r]:
                    j += 1
                else:
                    tr.append(fr[d])
                    cur.execute(""" INSERT INTO frame(num, emotion, time) VALUES(?, ?, ?)""", (nnn, thr[d], fr[d]))
                    d = r
                    j += 1
                    pos12 = cam.get(cv2.CAP_PROP_POS_MSEC)
                    pos1 = pos12 / 1000
                    tr.append(pos1)
            if y == 0:
                thr.pop(0)
            y += 1
        s.commit()
        out.write(c)
        # cv2.imshow('i', c)
finally:
    s.close()
    u = False
    audio.write_audiofile(to1 + "/video_out/output_vidd.mp3")
    input_video = ffmpeg.input(to1 + "/video_out/output_vid.avi")
    added_audio = ffmpeg.input(to1 + "/video_out/output_vidd.mp3").audio.filter('adelay', "1500|1500")
    (
        ffmpeg
            .concat(input_video, added_audio, v=1, a=1)
            .output(to1 + "/video_out/output_vidd.avi")
            .run(overwrite_output=True)
    )
    for num in tr:
        g1 += 1
        if g1 % 2 == 1:
            ffmpeg_extract_subclip(to1 + "/video_out/output_vidd.avi", num, tr[tr.index(num) + 1],
                                   targetname=to1 + "/video_out/vc" + str(p) + ".mp4")
            p += 1
    cam.release()
    out.release()
    cv2.destroyAllWindows()
# /Users/dimalavr1/Fiverr/order/Cont/sec.mp4
