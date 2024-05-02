import sys
import os

def get_object_aware_model(path):
    """ 
    'ObjectAwareModel' is a Yolov8 model that is distributed via the MobileSAMv2 github repo
    
    This routine is necessary because the torch weights were pickled with its environment, which messed up the import of the ObjectAwareModel class

    We also need to isolate this such that it will only be called in the case that we want to use the ObjectAwareModel class

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    current_dir = os.path.dirname(__file__)
    obj_detect_dir = os.path.join(current_dir, "object_detection")
    sys.path.insert(0, obj_detect_dir)

    from segment_everything.vendored.object_detection.ultralytics.prompt_mobilesamv2 import (
        ObjectAwareModel,
    )

    return ObjectAwareModel(path)