from fastapi import FastAPI, File, UploadFile
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import time
import datetime

import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json
import glob
import pandas as pd
import os
import ast
import time, copy
from collections import *
import gc
from torch.optim import lr_scheduler
from torch.cuda import amp
from tqdm.notebook import tqdm
from colorama import Fore, Back, Style
import albumentations
import albumentations.pytorch
import albumentations as A

from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import re
import random

import cv2
from matplotlib import pyplot as plt

pth = "/app/data/best_epoch-8.bin"


class Module:
    def __init__(self, device='cpu'):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2, pretrained=False)
        self.augs = albumentations.Compose([
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        self.device = device
        self.threshold = 0.99
        self.min_detections = 8

    def load_device(self, device):
        self.device = device

    def load_model(self, pth):
        self.model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))

    def load_image(self, pth):
        img = Image.open(pth)
        return img

    def video2frames(self, pth):
        vd = cv2.VideoCapture(pth)
        full_frames = int(vd.get(7))
        imgs = []
        for i in range(full_frames):
            _, frame = vd.read()
            imgs.append(frame)
        return imgs[::5]

    @torch.no_grad()
    def inference_image(self, img):
        self.model.eval()
        if self.augs:
            augmented = self.augs(image=np.asarray(img))
            img = augmented['image']
        img = img.to(self.device)
        preds = self.model([img])[0]
        cur = 0
        for i in range(len(preds['scores'])):
            if preds['scores'][i] < self.threshold:
                cur = i
                break
        for vl in preds:
            preds[vl] = preds[vl][:max(self.min_detections, cur)]
        return preds


api = Module()
api.load_model(pth)


def get_rock_size(boxes):
    # Бралось из расчёта, что # 10 см = 600 пикселей -> 1 мм = 0.17 пикселей
    sizes = []
    for box in boxes:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        roc_size_pixels = max(abs(x_max - x_min), abs(y_max - y_min))
        # 10 sm = 600 pixels -> 1 mm = 0.17 pixels
        roc_size_mm = roc_size_pixels * 1.7

        if 0 <= roc_size_mm <= 40:
            sizes.append(7)
        elif 40 < roc_size_mm <= 70:
            sizes.append(6)
        elif 70 < roc_size_mm <= 80:
            sizes.append(5)
        elif 80 < roc_size_mm <= 100:
            sizes.append(4)
        elif 100 < roc_size_mm <= 150:
            sizes.append(3)
        elif 150 < roc_size_mm <= 250:
            sizes.append(2)
        elif roc_size_mm > 250:
            sizes.append(1)
    return sizes


tags_metadata = [
    {
        "name": "image",
        "description": "Обработка изображения .jpg",
    },
    {
        "name": "video",
        "description": "Обработка видео .mp4 и получение статистики",
    },
    {
        "name": "stream",
        "description": "Получение кадра из видеопотока и его обработка",
    },
    {
        "name": "streamSocket",
        "description": "Получение сокета с информацией о видеопотоке в реальном времени",
    },
]

app = FastAPI(
    title="Хозяин руды - Проигрышный вариант",
    description="Добро пожаловать в документацию API решения команды 'Проигрышный вариант'.",
    version="1.0.1",
    contact={
        "name": "Команда 'Проигрышный вариант'",
        "email": "mail@artdorokhin.ru",
    },
    openapi_tags=tags_metadata)

origins = [
    "http://localhost",
    "https://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Welcome to the Bezproigrishniy Variant": "API"}


@app.post("/image/", tags=["image"])
def image_pred(file: bytes = File()):
    try:
        d = datetime.datetime.now()
        unixtime = str(time.mktime(d.timetuple()))
        filename = '/app/data/' + unixtime + '.jpg'
        f = open(filename, 'wb')
        f.write(file)
        f.close()

        img = api.load_image(filename)
        ans = api.inference_image(img)
        ans['boxes'] = ans['boxes'].tolist()
        ans['scores'] = ans['scores'].tolist()
        ans['sizes'] = get_rock_size(ans['boxes'])

        return ans
    except Exception as e:
        return {"error": str(e)}


@app.post("/video/", tags=["video"])
def video_pred(file: bytes = File()):
    try:
        d = datetime.datetime.now()
        unixtime = str(time.mktime(d.timetuple()))
        filename = '/app/data/' + unixtime + '.mp4'
        f = open(filename, 'wb')
        f.write(file)
        f.close()

        imgs = api.video2frames(filename)
        anss = []
        for img in imgs:
            ans = api.inference_image(img)

            ans['boxes'] = ans['boxes'].tolist()
            ans['scores'] = ans['scores'].tolist()

            ans['sizes'] = get_rock_size(ans['boxes'])
            anss.append(ans)
        return anss
    except Exception as e:
        return {"error": str(e)}
