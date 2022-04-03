import os
import time
import datetime
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from matplotlib import pyplot as plt

import serial
import os
import cv2
#starts the webcam, uses it as video source
camera = cv2.VideoCapture(0) #uses webcam for video

NUM_CLASSES = 2
THRESHOLD = 0.8
CLASSES = ['Litter']

transform = T.Compose([
    T.ToTensor()
])

import argparse
import datetime
from pathlib import Path
import time
import os
import torchvision
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

try:
    arduino = serial.Serial(port='COM7', baudrate=115200, timeout=.1)
except Exception as err:
    print(err)

def write_read(x):
    try:
        arduino.write(bytes(x, 'utf-8'))
        time.sleep(0.05)
        data = arduino.readline()
    except:
        data = -1
    return data



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_instance_segmentation_model(
        num_classes, model_name='maskrcnn_resnet50_fpn'):

    model = torchvision.models.detection.__dict__[model_name](
        pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_name.startswith('mask') or model_name.startswith('efficientnet'):

        in_features_mask = \
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
    return model

model = get_instance_segmentation_model(NUM_CLASSES)

model.eval()
ind_count=0
while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = camera.read()
    frame_op = frame.copy()
    ind_count+=1

    cv2.imshow('object detection', cv2.resize(frame_op, (800, 600)))
    #display every 5 seconds
    #do inference

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if ind_count%120 ==0 :
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        # read an image
        #im = Image.open("000009.jpg").convert('RGB')  # get it from frame

        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)
        outputs = model(img)

        plt.figure(figsize=(16,10))
        img = np.array(im)[:,:,1]
        plt.imshow(im)
        ax = plt.gca()
        keep = outputs[0]['scores'].detach().numpy() > THRESHOLD
        prob = outputs[0]['scores'].detach()[keep]
        labels = outputs[0]['labels'].detach().numpy()[keep].tolist()
        masks = outputs[0]['masks'].detach().numpy()[keep]

        try:
            max_prob = [int(np.argmax(prob))]
        except Exception as err:
            print(err)
            max_prob = 0
        
        masking = np.zeros((1,)+img.shape)

        for j, i in enumerate(outputs[0]['boxes'].detach().numpy()[max_prob].tolist()):
            p = prob[j]
            masking += masks[j]
            ax.add_patch(plt.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1],
                                        fill=False, color='r', linewidth=3))
            cl = int(labels[j])-1
            text = f'{CLASSES[cl]}: {p:0.2f}'
            ax.text(i[0], i[1], text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))


        imagines = np.array(im)
        imagines[:,:,0] = imagines[:,:,0] + masking[0,:,:]*100
        plt.imshow(imagines)
        plt.axis('off')
        plt.show()

        # writing to arduino 
        num = 0 # default todo
        if max_prob > 0.75:
            num = '3'
        elif max_prob > 0.45 and max_prob<=0.75:
            num = '2'
        elif max_prob > 0.25 and max_prob<=0.45:
            num = '1'
        else:
            num = '1'

        print('Num value to be written is ',num)
        value = write_read(num)
        print('Value returned by arduino is ',value)


"""

plt.figure(figsize=(16,10))
plt.imshow(masking[0,:,:])
plt.axis('off')
plt.show()
"""