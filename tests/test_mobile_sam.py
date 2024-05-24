from skimage import data
from segment_everything.weights_helper import get_weights_path
from segment_everything.weights_helper import create_mobile_sam_model
from segment_everything.prompt_generator import YoloDetector
from segment_everything.detect_and_segment import segment_from_bbox
from segment_everything.vendored.mobilesamv2 import SamPredictor as SamPredictorV2

def test_mobile_sam():
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
    
    model = create_mobile_sam_model()

    predictor = SamPredictorV2(model)
    predictor.set_image(image)

    sam_masks = segment_from_bbox(bounding_boxes, predictor, model, device)

    #from segment_everything.napari_helper import to_napari
    #to_napari(image, sam_masks) 
    #input("Press enter to continue...")
       

test_mobile_sam()