from skimage import data
from segment_everything.weights_helper import get_weights_path
from segment_everything.object_detectors.yolo_detector import YoloDetector
from segment_everything.mask_detectors.mobilesam import mobilesam_detector

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

    print("Detected {} objects".format(len(bounding_boxes)))

    # assert that 8 boxes are detected
    assert len(bounding_boxes) == 8

    
    detector = mobilesam_detector(model_type="MobileSamV2", device=device)
    detector.set_image(image)
    sam_masks = detector.segment_boxes(bounding_boxes)

    print("Area of first mask: ", sam_masks[0]["area"])

    # assert the area is correct
    assert sam_masks[0]["area"] == 54563.0

    #from segment_everything.napari_helper import to_napari
    #to_napari(image, sam_masks) 
    #input("Press enter to continue...")
       

test_mobile_sam()