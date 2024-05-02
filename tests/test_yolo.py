from skimage import data
from segment_everything.weights_helper import get_weights_path
from segment_everything.prompt_generator import YoloDetector

def test_bounding_box_detection():
    image = data.coffee()

    conf = 0.5
    iou = 0.5
    imgsz = 512
    device = "cpu"
    max_det = 100

    model = YoloDetector(
        str(get_weights_path("ObjectAwareModel")), "ObjectAwareModelFromMobileSamV2", device=device
    )
    bounding_boxes = model.get_bounding_boxes(
        image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det
    )
