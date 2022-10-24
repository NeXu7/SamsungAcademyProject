from stardist.models import StarDist2D
import torch
from torchvision import models, transforms
import numpy as np
from torchvision.transforms.functional import rotate
import cv2 as cv
from csbdeep.utils import normalize


def get_classifier():
    model = models.densenet121(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier.in_features, 500),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(500, 5),
    )
    return model


def get_detector():
    model_stardist = StarDist2D.from_pretrained("2D_versatile_he")
    return model_stardist


class RoiWorker:
    def __init__(self, model_stardist, model_classifier):
        self.model_stardist = model_stardist
        self.model_classifier = model_classifier
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mean = [0.65, 0.49, 0.64]
        std = [0.16, 0.20, 0.20]
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean, std),
            ]
        )
        self.model_classifier.to(self.device)

    def predict_stardist(self, image):
        print("Detections predicted")
        _, he_2 = self.model_stardist.predict_instances(normalize(image))
        return he_2

    def predict_class(self, image, detections):
        predicts = []
        for coord in detections["coord"]:
            try:
                coord = np.rot90(coord, k=-1).astype(np.int64)
                x, y, w, h = np.array(cv.boundingRect(coord)) + (-5, -5, 10, 10)
                small_image = image[y : y + h, x : x + w].copy()
                small_mask = np.zeros_like(small_image)[:, :, 0].astype(np.uint8)
                param = np.array(small_mask.shape) // 2
                cv.circle(small_mask, param, min(param + 1), 255, -1)
                small_image[small_mask == 0] = [0, 0, 0]
                small_image = self.resize_img(small_image)
                # if small_image is not None:
                small_image = self.image_transforms(small_image)
                images = [rotate(small_image, angle=i * 90) for i in range(2)]
                images = [torch.unsqueeze(img, dim=0) for img in images]
                images = torch.cat(images)
                predict = self.densenet_predict(images)
                predicts.append(predict)
            except:
                predicts.append(None)
        return predicts

    def predict(self, image):
        detections = self.predict_stardist(image)
        labels = self.predict_class(image, detections)
        json_form = self.create_json(detections["coord"], labels)
        return json_form

    @staticmethod
    def resize_img(img, shape=56):
        img_w, img_h, _ = img.shape
        vert_add = 0
        side_add = 0
        if img_h < shape:
            side_add = (shape - img_h) // 2
        elif img_h > shape:
            side_crop = (img_h - shape) // 2
            img = img[:, side_crop:-side_crop]
        if img_w < shape:
            vert_add = (shape - img_w) // 2
        elif img_w > shape:
            vert_crop = (img_w - shape) // 2
            img = img[vert_crop:-vert_crop]
        img = cv.copyMakeBorder(
            img,
            vert_add,
            vert_add,
            side_add,
            side_add,
            cv.BORDER_CONSTANT,
            None,
            [0, 0, 0],
        )
        return img

    def densenet_predict(self, batch):
        with torch.no_grad():
            batch = batch.to(self.device)
            predict = self.model_classifier(batch)
            predict = self.softmax(predict).data.cpu().numpy()
            return int(np.argmax(predict.mean(0)))

    @staticmethod
    def create_json(detections, labels):
        json_form = []
        for label, detection in zip(labels, detections):
            if label is not None:
                if label != 4 or label != 0:
                    detection = np.rot90(detection, k=-1)
                    detection = detection.tolist()
                    json_form.append({"label": label, "detection": detection})

        return json_form


def inference(image):
    MODEL_PATH = "models/final_model.pt"
    model_densenet = get_classifier()
    model_densenet.load_state_dict(torch.load(MODEL_PATH))
    model_densenet.eval()
    model_stardist = StarDist2D.from_pretrained("2D_versatile_he")
    roi_worker = RoiWorker(model_stardist, model_densenet)
    predict = roi_worker.predict(image)
    return predict
