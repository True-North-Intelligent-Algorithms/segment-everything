from typing import Iterable, List, Dict, Any, Optional
import numpy as np
from segment_everything.mask_detectors.base_mask_detector import BaseMaskDetector


class microsam_detector(BaseMaskDetector):
    """Adapter for MicroSAM-style segmentation.

    Exposes the same minimal API as the mobilesam_detector:
      - set_image(img)
      - segment_boxes(boxes) -> list[ann dict]

    Uses `micro_sam.sam_annotator._state.AnnotatorState` and
    `micro_sam.prompt_based_segmentation.segment_from_box` internally.
    """

    def __init__(self, model_type: str = "vit_b_lm", device: Optional[str] = None):
        # Use model_type as the BaseDetector model_path identifier
        super().__init__(model_type, trainable=False)
        self.model_type = model_type
        self.device = device
        self._state = None
        self._image = None

    def set_image(self, image: Any):
        """Initialize the MicroSAM predictor and embeddings for the given image."""
        try:
            from micro_sam.sam_annotator._state import AnnotatorState
        except Exception as e:
            raise RuntimeError("micro_sam is required for microsam_detector") from e

        state = AnnotatorState()
        state.reset_state()
        state.initialize_predictor(
            image,
            model_type=self.model_type,
            ndim=2,
        )
        self._state = state
        self._image = image

    def segment_boxes(self, boxes: Iterable) -> List[Dict[str, Any]]:
        """Segment a list of boxes using the MicroSAM predictor.

        Args:
            boxes: iterable of [x1,y1,x2,y2] boxes (numpy arrays or lists)

        Returns:
            list of annotation dicts with keys similar to the MobileSAM helper
        """
        if self._state is None:
            raise RuntimeError("set_image must be called before segment_boxes")

        try:
            from micro_sam import prompt_based_segmentation
        except Exception as e:
            raise RuntimeError("micro_sam is required for microsam_detector") from e

        anns = []
        for bbox in boxes:
            bbox_np = np.array(bbox)
            prediction = prompt_based_segmentation.segment_from_box(
                self._state.predictor, bbox_np, image_embeddings=self._state.image_embeddings
            )
            prediction = np.squeeze(prediction)
            ann = {
                "segmentation": prediction,
                "area": float(prediction.sum()),
                "predicted_iou": None,
                "stability_score": None,
                "prompt_bbox": bbox_np,
            }
            if prediction.max() < 1:
                continue
            anns.append(ann)

        return anns
