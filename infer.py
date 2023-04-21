import json
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

import cv2
import base64
import numpy as np
import requests
import time

upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])
streamURL = "https://192.168.34.98:8080/video"
#video = cv2.VideoCapture(0)
video = cv2.VideoCapture(streamURL)

def infer():
    ret, img = video.read()

    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

while 1:
    if(cv2.waitKey(1) == ord('q')):
        break

    start = time.time()

    image = infer()

    cv2.imshow('image', image)

    print((1/(time.time()-start)), " fps")

video.release()
cv2.destroyAllWindows()