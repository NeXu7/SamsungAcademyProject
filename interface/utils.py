import os

import numpy as np
import cv2 as cv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import slideio as sio
from PIL import Image


class ImgData(Dataset):
    def __init__(self, file_names, file_dir):
        self.file_names = file_names
        self.file_dir = file_dir
        mean = [0.65, 0.49, 0.64]
        std = [0.16, 0.20, 0.20]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.file_dir, self.file_names[item]))
        images = [img.rotate(45 * i) for i in range(8)]
        images = [self.transform(img) for img in images]
        images = torch.FloatTensor(images)
        return images


class SlideData(Dataset):
    def __init__(self, slide, contours):
        self.slide = slide
        self.contours = contours
        mean = [0.65, 0.49, 0.64]
        std = [0.16, 0.20, 0.20]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, item):
        img = self.cell_mask(self.contours.iloc[item], self.slide, pix_around=10)
        h, w, _ = img.shape
        top = (256 - h) // 2
        left = (256 - w) // 2
        img = cv.copyMakeBorder(img, top, top, left, left, cv.BORDER_CONSTANT, None, [0, 0, 0])
        img_batch = [self.transform_img(img, i * 45, self.transform) for i in range(8)]
        img_batch = torch.FloatTensor(img_batch)
        return img_batch

    @staticmethod
    def cell_mask(cell_cont, image, pix_around=10, x_offset=0, y_offset=0):
        cell_cont = np.array(cell_cont)
        x_img, y_img, w, h = cv.boundingRect(cell_cont)
        x = x_img - pix_around - x_offset
        y = y_img - pix_around - y_offset
        w = w + 2 * pix_around
        h = h + 2 * pix_around
        cell_img = image[y:y + h, x:x + w].copy()
        cell_cont = cell_cont - [x_img - pix_around, y_img - pix_around]
        mask = np.zeros(cell_img.shape[:2])
        mask = cv.drawContours(mask, cell_cont, -1, (1), -1) + cv.drawContours(mask, cell_cont, -1, (1),
                                                                               pix_around * 2)
        mask = mask >= 1
        cell_img[np.logical_not(mask)] = [0, 0, 0]
        return cell_img

    @staticmethod
    def transform_img(img, angle, transform):
        img = cv.rotate(img, angle)
        img = transform(img)
        return img


def get_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.classifier.in_features, 500),
            torch.nn.Linear(500, 5)
        )
        model.load_state_dict(torch.load("models/densenet121.pt", map_location=device))
        model.eval()
        return model

    if model_name == "densenet201":
        model = models.densenet201(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.classifier.in_features, 500),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(500, 5)
        )
        model.load_state_dict(torch.load("models/densenet201.pt", map_location=device))
        model.eval()
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features,
                                              out_features=5)
        model.load_state_dict(torch.load("models/efficientnet_b0.pt", map_location=device))
        model.eval()
        return model


def classify_cell(model, data, slide, contours):
    data = SlideData(slide=slide, contours=contours)
    softmax = torch.nn.Softmax(dim=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    voted_predict = []
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            predict = softmax(model(batch))
            predict = predict.data.cpu().numpy()
            voted_predict.append(predict.mean(0))
    return np.array(voted_predict)


def get_data(slide_path=None, counter_path=None, img_folder_path=None):
    if slide_path is not None and counter_path is not None:
        data = pd.read_json(counter_path)
        size = data.iloc[0].astype(np.int64)
        counters = data.iloc[1:]
        slide = sio.open_slide(slide_path, 'SVS')
        scene = slide.get_scene(0)
        slide = scene.read_block(size)
        return SlideData(slide=slide, contours=counters)

    if img_folder_path is not None:
        img_names = os.listdir(img_folder_path)
        return ImgData(file_names=img_names, file_dir=img_names)
