"""
DEPRECATED module shim.

`segment_everything.prompt_generator` was replaced by separate classes in
`segment_everything.object_detectors` (base_detector, yolo_detector,
`rcnn_detector`). To maintain compatibility for external code that still
imports from the old path, this file re-exports the new classes and emits a
DeprecationWarning.

This shim will be removed in a future release; please update imports to:

    from segment_everything.object_detectors.yolo_detector import YoloDetector
    from segment_everything.object_detectors.rcnn_detector import RcnnDetector
    from segment_everything.object_detectors.base_detector import BaseDetector

"""
from warnings import warn

warn(
    "segment_everything.prompt_generator is deprecated — import detectors from "
    "segment_everything.object_detectors instead",
    DeprecationWarning,
)

from segment_everything.object_detectors.base_detector import BaseDetector
from segment_everything.object_detectors.yolo_detector import YoloDetector
from segment_everything.object_detectors.rcnn_detector import RcnnDetector

__all__ = ["BaseDetector", "YoloDetector", "RcnnDetector"]
