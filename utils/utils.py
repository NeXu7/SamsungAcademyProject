import os

import streamlit
from stqdm import stqdm
import numpy as np
import cv2 as cv
import pandas as pd
from scipy import ndimage
import torch
from torch.utils.data import Dataset
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
        images = [img.rotate(i * 180) for i in range(2)]
        images = [torch.unsqueeze(self.transform(img), dim=0) for img in images]
        images = torch.cat(images)
        return images


class SlideData(Dataset):
    def __init__(self, slide, contours, x_offset, y_offset):
        self.slide = slide
        self.contours = contours
        mean = [0.65, 0.49, 0.64]
        std = [0.16, 0.20, 0.20]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean, std)
        ])
        self.x_offset, self.y_offset = x_offset, y_offset

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, item):
        img = self.cell_mask(self.contours[item], pix_around=10)
        h, w, _ = img.shape
        top = (256 - h) // 2
        left = (256 - w) // 2
        img = cv.copyMakeBorder(img, top, top, left, left, cv.BORDER_CONSTANT, None, [0, 0, 0])
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        # images = [img.rotate(i * 180) for i in range(2)]
        # images = [torch.unsqueeze(self.transform(img), dim=0) for img in images]
        # images = torch.cat(images)
        streamlit.image(img)
        img_batch = [torch.unsqueeze(self.transform_img(img, i*45, self.transform),
                                     dim=0) for i in range(8)]
        img_batch = torch.cat(img_batch)
        return img_batch

    def cell_mask(self, cell_cont, pix_around=10):
        cell_cont = np.array(cell_cont)
        x_img, y_img, w, h = cv.boundingRect(cell_cont)
        x = x_img - pix_around - self.x_offset
        y = y_img - pix_around - self.y_offset
        w = w + 2 * pix_around
        h = h + 2 * pix_around
        cell_img = self.slide[y:y + h, x:x + w].copy()
        cell_cont = cell_cont - [x_img - pix_around, y_img - pix_around]
        mask = np.zeros(cell_img.shape[:2])
        mask = cv.drawContours(mask, cell_cont, -1, (1), -1) + cv.drawContours(mask, cell_cont, -1, (1),
                                                                               pix_around * 2)
        mask = mask >= 1
        cell_img[np.logical_not(mask)] = [0, 0, 0]

        return cell_img

    @staticmethod
    def transform_img(img, angle, transform):
        img = ndimage.rotate(img, angle)
        img = transform(img)
        return img


def contour_ext(data):
    data = data.dropna(axis=0)
    measurements = []
    for i in range(len(data)):
        measurements.append(np.array(data['nucleusGeometry'].iloc[i]['coordinates'], dtype=np.int32))
    return measurements


def get_model(model_name):
    '''

    :param model_name: name of model to be predictor
    :return: prepared model
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.classifier.in_features, 500),
            torch.nn.Dropout(p=0.2),
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


def classify_cell(model, data):
    softmax = torch.nn.Softmax(dim=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    voted_predict = []
    with torch.no_grad():
        for batch in stqdm(data, desc=f"Device: {device}. Progress"):
            batch = batch.to(device)
            predict = softmax(model(batch))
            predict = predict.data.cpu().numpy()
            voted_predict.append(predict.mean(0))
    return np.array(voted_predict)


def get_data(slide_path=None, contour_data=None, img_folder_path=None):
    """
    create dataloader for prediction
    :param slide_path: path to the WSI
    :param contour_data: cells contour predicted with StarDirst in QuPath
    :param img_folder_path: folder where to save images
    :return: dataloader with images to predict
    """
    if slide_path is not None and contour_data is not None:
        X = []
        Y = []
        for cont in contour_data:
            x, y, w, h = cv.boundingRect(cont)
            X.append(x)
            Y.append(y)
        size = (np.min(X) - 200, np.min(Y) - 200, np.max(X) - np.min(X) + 400, np.max(Y) - np.min(Y) + 400)
        slide = sio.open_slide(slide_path, 'SVS')
        scene = slide.get_scene(0)
        slide = scene.read_block(size)
        return SlideData(slide=slide, contours=contour_data,
                         x_offset=size[0], y_offset=size[1])

    if img_folder_path is not None:
        img_names = [name for name in os.listdir(img_folder_path) if ".png" in name]
        return ImgData(file_names=img_names, file_dir=img_folder_path)


def make_predict_file(mask, predict):
    """
    Construct file with predictions for detected cells
    :param mask:
    :param predict:
    :return:
    """
    ccd = {4: 16711680, 0: 255, 2: 16776960, 3: 16737280, 1: 6618880}
    for i in range(len(mask)):
        if predict[i, 0] > 0.3:
            mask.loc[i, 'properties']['classification'] = {'name': 0,
                                                           'colorRGB': ccd[0]
                                                           }
            mask.loc[i, 'properties']['measurements'] = [{'name': 'Grade', 'value': 0}]
        else:
            cell_class = int(np.argmax(predict[i]))
            mask.loc[i, 'properties']['classification'] = {'name': cell_class,
                                                           'colorRGB': ccd[cell_class]
                                                           }
            mask.loc[i, 'properties']['measurements'] = [{'name': 'Grade', 'value': cell_class}]
    return mask
