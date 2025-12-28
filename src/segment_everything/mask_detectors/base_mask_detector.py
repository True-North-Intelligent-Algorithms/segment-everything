from typing import Iterable, List, Dict, Any
from segment_everything.object_detectors.base_object_detector import BaseDetector


class BaseMaskDetector(BaseDetector):
    """Base class for mask/segmentation detectors.

    Mask detectors (SAM-style) should derive from this class and implement
    the two small APIs used by the rest of the codebase:
      - set_image(image)
      - segment_boxes(boxes, **kwargs) -> list[annotation dicts]

    This file lives under `mask_detectors` to make the mask-specific API
    colocated with adapter implementations.
    """

    def set_image(self, image: Any):
        raise NotImplementedError()

    def segment_boxes(self, boxes: Iterable, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError()
