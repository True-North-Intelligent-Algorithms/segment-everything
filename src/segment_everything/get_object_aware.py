import sys
import os

def get_object_aware_model(model_path):
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
    obj_detect_dir = os.path.join(current_dir, "vendored", "object_detection")
    sys.path.insert(0, obj_detect_dir)

    from segment_everything.vendored.object_detection.ultralytics.prompt_mobilesamv2 import ObjectAwareModel

    return ObjectAwareModel(model_path)