#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:58:14 2024

@author: ian
"""
import cv2
import os, sys
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import ToTensor
from torchvision.ops import nms
import torch

from segment_everything.get_object_aware import get_object_aware_model

class BaseDetector:
    def __init__(self, model_path, trainable=False):
        self.model_path = model_path
        self.model_name = self.model_path.split("/")[-1]
        self.trainable = trainable

    def train(self, training_data):
        raise NotImplementedError()

    def predict(self, image_data):
        raise NotImplementedError()


class YoloDetector(BaseDetector):
    def __init__(self, model_path, model_type, device, trainable=False):
        super().__init__(model_path)
        self.model_type = model_type

        if (model_type == "ObjectAwareModelFromMobileSamV2"):
            self.model = get_object_aware_model(model_path) #ObjectAwareModel(model_path)
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

        self.device = device

    def train(self):
        print(
            "YOLO detector is not yet trainable, use RcnnDetector for training"
        )

    def get_results(
        self,
        image_data,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
        max_det=400,
    ):
        """
        Runs YOLO and returns the YOLO results 

        Parameters
        ----------
        image : numpy.ndarray
            A 2D-image in grayscale or RGB.
        imgsz : INT, optional
            Size of the input image. The default is 1024.
        conf : FLOAT, optional
            Confidence threshold for the bounding boxes. Lower means more boxes will be detected. The default is 0.4.
        iou : FLOAT, optional
            Threshold for how many intersecting bounding boxes should be allowed. Lower means fewer intersecting boxes will be returned. The default is 0.9.
        max_det : INT, optional
            Maximum number of detections that will be returned. The default is 400.

        Returns
        -------
        obj_results: YOLO results objects
        """

        image_cv2 = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        obj_results = self.model.predict(
            image_cv2,
            device=self.device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )

        return obj_results
    
    def get_bounding_boxes(
        self,
        image_data,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
        max_det=400,
    ):
        """
        Generates a series of bounding boxes in xyxy-format from an image, using the YOLOv8 ObjectAwareModel.

        Parameters
        ----------
        image : numpy.ndarray
            A 2D-image in grayscale or RGB.
        imgsz : INT, optional
            Size of the input image. The default is 1024.
        conf : FLOAT, optional
            Confidence threshold for the bounding boxes. Lower means more boxes will be detected. The default is 0.4.
        iou : FLOAT, optional
            Threshold for how many intersecting bounding boxes should be allowed. Lower means fewer intersecting boxes will be returned. The default is 0.9.
        max_det : INT, optional
            Maximum number of detections that will be returned. The default is 400.

        Returns
        -------
        bounding_boxes : numpy.ndarray
            An array of boxes in xyxy-coordinates.
        """

        print("Predicting bounding boxes for image data")
        obj_results = self.get_results(image_data, retina_masks, imgsz, conf, iou, max_det) 

        return obj_results[0].boxes.xyxy.cpu().numpy()
  
    def __str__(self):
        s = f"\n{'Model':<10}: {self.model_name}\n"
        s += f"{'Type':<10}: {str(self.model_type)}\n"
        s += f"{'Trainable':<10}: {str(self.trainable)}"
        return s


class RcnnDetector(BaseDetector):
    def __init__(self, model_path, device, trainable=True):
        super().__init__(model_path, trainable)
        self.model_type = "FasterRCNN"
        if device == "mps":
            device = "cpu"
        self.device = device
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            box_detections_per_img=500,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))

    def train(self, training_data):
        if self.trainable:
            print("Training model")
            print(self.model_path)
            print(training_data)

    def _get_transform(self, train):
        from torchvision.transforms import v2 as T

        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    @torch.inference_mode()
    def get_bounding_boxes(self, image_data, conf=0.5, iou=0.2):
        image_cv2 = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        print("Predicting bounding boxes for image data")
        convert_tensor = ToTensor()
        eval_transform = self._get_transform(train=False)
        tensor_image = convert_tensor(image_cv2)
        x = eval_transform(tensor_image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(self.device)
        self.model.eval()
        predictions = self.model([x])
        pred = predictions[0]
        #    print(pred)
        idx_after = nms(pred["boxes"], pred["scores"], iou_threshold=iou)
        pred_boxes = pred["boxes"][idx_after]
        pred_scores = pred["scores"][idx_after]
        pred_boxes_conf = pred_boxes[pred_scores > conf]
        return pred_boxes_conf.cpu().numpy()

    def __str__(self):
        s = f"\n{'Model':<10}: {self.model_name}\n"
        s += f"{'Type':<10}: {str(self.model_type)}\n"
        s += f"{'Trainable':<10}: {str(self.trainable)}"
        return s
