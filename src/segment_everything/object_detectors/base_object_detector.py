class BaseDetector:
    """Generic base detector.

    This class provides minimal bookkeeping (model path/name and trainable flag).
    Specialized detector types (object vs mask) should derive from this class and
    implement the concrete API methods documented below.
    """

    def __init__(self, model_path: str, trainable: bool = False):
        self.model_path = model_path
        self.model_name = self.model_path.split("/")[-1] if model_path else ""
        self.trainable = trainable

    def train(self, training_data):
        """Train the detector (optional).

        Subclasses that support training should override this method.
        """
        raise NotImplementedError()

    def predict(self, image_data):
        """Generic predict entrypoint.

        Subclasses should implement either `get_results` (object detectors) or
        `segment_boxes`/`set_image` (mask detectors) depending on their API.
        """
        raise NotImplementedError()


class BaseObjectDetector(BaseDetector):
    """Base class for object/bounding-box detectors.

    Expected concrete methods:
      - get_results(image_data, **kwargs) -> framework-specific result object
      - get_bounding_boxes(image_data, **kwargs) -> numpy array of [x1,y1,x2,y2]
    """

    def get_results(self, image_data, **kwargs):
        raise NotImplementedError()

    def get_bounding_boxes(self, image_data, **kwargs):
        raise NotImplementedError()

    def get_microsam_bboxes(self, image_data, **kwargs):
        """
        Return bounding boxes formatted for MicroSAM: list of [x1, y1, x2, y2].

        Subclasses that wrap specific detectors should implement this convenience
        method so notebooks and downstream code can call a single API to get
        boxes ready for segmentation.
        """
        raise NotImplementedError()

    def get_napari_bboxes(self, image_data, **kwargs):
        """
        Return bounding boxes formatted for Napari: list of [[y1,x1],[y2,x2]] per box.

        This helper provides a common rectangle format used when adding shapes to
        Napari viewers.
        """
        raise NotImplementedError()


# Note: mask-specific base class lives under `mask_detectors/base_mask_detector.py`
# to keep mask APIs colocated with mask adapter implementations.
