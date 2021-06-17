import cv2
import numpy as np
from playsound import playsound

net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg','yolov4-tiny.weights')
classes = []
with open('coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

path = 'traffic.mp4'
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame,(900,520))
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        while True:
            ht, wt, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            last_layer = net.getUnconnectedOutLayersNames()
            layer_out = net.forward(last_layer)
            boxes = []
            confidences = []
            cls_ids = []
            for output in layer_out:
                for detection in output:
                    score = detection[5:]
                    clsid = np.argmax(score)
                    conf = score[clsid]
                    if conf > 0.5:
                        centreX = int(detection[0] * wt)
                        centreY = int(detection[1] * ht)
                        w = int(detection[2] * wt)
                        h = int(detection[3] * ht)
                        x = int(centreX - w / 2)
                        y = int(centreY - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append((float(conf)))
                        cls_ids.append(clsid)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .3, .2)
            colors = np.random.uniform(0, 255, size=(len(boxes), 2))
            cv2.line(frame, (360, 125), (360, 520), (155, 155, 155), 2)
            cv2.line(frame, (540, 125), (540, 520), (155, 155, 155), 2)
            cv2.line(frame, (360, 125), (540, 125), (155, 155, 155), 2)
            #cv2.line(frame, (790, 380), (900, 440), (255, 255, 0), 2)
            try:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[cls_ids[i]])
                    if (label == 'car') or (label == 'bus') or (label == 'truck'):
                        #dist = round((1 - (detection[3] - detection[1])) ** 4, 1)
                        confidence = str(round(confidences[i] * 100, 1)) + '%'
                        color = colors[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label.upper() + "-" + confidence, (x, y - 5), font, 0.7, (0, 255, 255), 1)
                        if (x < 360) and (x+w > 540) and (y>125):

                            cv2.putText(frame,'Warning!', (centreX-30, centreY),font,1.5, (0, 0, 255), 2)
                            playsound('beep.wav')
                    ret, jpeg = cv2.imencode('.jpg', frame)


                return jpeg.tobytes()

            except:
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
