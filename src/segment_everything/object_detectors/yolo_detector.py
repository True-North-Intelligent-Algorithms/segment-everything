import cv2
import sys
import os
import numpy as np
from segment_everything.object_detectors.base_object_detector import BaseObjectDetector
from segment_everything.vendored.object_detection.ultralytics.prompt_mobilesamv2 import ObjectAwareModel

class YoloDetector(BaseObjectDetector):

    def __init__(self, model_path, model_type, device, trainable=False):
        super().__init__(model_path)
        self.model_type = model_type

        if (model_type == "ObjectAwareModelFromMobileSamV2"):
            self.model = self.get_object_aware_model(model_path)
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
        print("Predicting bounding boxes for image data")
        obj_results = self.get_results(image_data, retina_masks, imgsz, conf, iou, max_det)
        return obj_results[0].boxes.xyxy.cpu().numpy()

    def get_object_aware_model(self, model_path):
        """
        Returns an instance of ObjectAwareModel (a YOLOv8 model) distributed via the MobileSAMv2 repo.

        This routine is necessary because the torch weights were pickled with their environment,
        which complicates importing the ObjectAwareModel class. We isolate this import so it is
        only called when ObjectAwareModel is needed.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            ObjectAwareModel: Instantiated YOLOv8 ObjectAwareModel.
        """
        # Add ultralytics object detection directory to sys.path for import
        current_dir = os.path.dirname(__file__)
        obj_detect_dir = os.path.join(current_dir, "..", "vendored", "object_detection")
        sys.path.insert(0, obj_detect_dir)

        return ObjectAwareModel(model_path)

    def get_microsam_bboxes(self, image_data, **kwargs):
        """
        Return bounding boxes formatted for MicroSAM: list of [x1, y1, x2, y2].
        """
        bboxes = self.get_bounding_boxes(image_data, **kwargs)

        bboxes_microsam = []
        for box in bboxes:
            bbox_microsam = [box[1], box[0], box[3], box[2]]
            bboxes_microsam.append(bbox_microsam)
        return bboxes_microsam

    def get_napari_bboxes(self, image_data, **kwargs):
        """
        Return bounding boxes formatted for Napari: list of [[y1,x1],[y2,x2]] per box.
        """
        bboxes = self.get_bounding_boxes(image_data, **kwargs)

        napari_boxes = []
        for box in bboxes:
            bbox_napari = [[box[1], box[0]], [box[3], box[2]]]
            napari_boxes.append(bbox_napari)
        return napari_boxes

    def __str__(self):
        s = f"\n{'Model':<10}: {self.model_name}\n"
        s += f"{'Type':<10}: {str(self.model_type)}\n"
        s += f"{'Trainable':<10}: {str(self.trainable)}"
        return s
