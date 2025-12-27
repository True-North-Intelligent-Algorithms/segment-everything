import cv2
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import ToTensor
from torchvision.ops import nms
import torch
from segment_everything.object_detectors.base_detector import BaseDetector

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
        x = x[:3, ...].to(self.device)
        self.model.eval()
        predictions = self.model([x])
        pred = predictions[0]
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
