import re
import numpy as np

from skimage.measure import label, regionprops

class StackedLabels:

    def __init__(self, mask_list=None, image = None, label_image=None):
        self.image = image

        if image is not None and image.ndim == 2:
            self.image = np.stack([self.image, self.image, self.image], axis=-1)
        
        self.label_image = label_image

        if mask_list is None:
            self.mask_list = []
        else:
            self.mask_list = mask_list

    @staticmethod
    def create_mask(segmentation, image=None):
        mask = {}
        mask['segmentation'] = segmentation
        y, x = np.where(segmentation)
        mask['indexes'] = [y, x]
        mask['point_coords'] = [[np.mean(x), np.mean(y)]]
        mask['prompt_bbox'] = StackedLabels.get_bbox(segmentation)
        mask['area'] = np.sum(segmentation)
        mask['predicted_iou'] = 1
        mask['stability_score'] = 1
        if image is not None:
            mask['image'] = image
        return mask
    
    @staticmethod    
    def get_bbox(segmentation):
        y, x = np.where(segmentation > 0)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return [x_min, y_min, x_max, y_max]

    @classmethod
    def from_2d_label_image(cls, label_image, image, relabel=True):
        if (relabel):
            label_image = label(label_image)

        mask_list = []
        for region in regionprops(label_image):
            segmentation = np.zeros_like(label_image, dtype=bool)
            segmentation[label_image == region.label] = True
            mask = cls.create_mask(segmentation, image)
            mask_list.append(mask)

        return cls(mask_list, image, label_image)

    def add_segmentation(self, segmentation):
        stacked_label = self.create_mask(segmentation)
        self.mask_list.append(stacked_label)

    def add_background_results(self, num_background_results=1):

        for i in range(num_background_results):
            background = (self.label_image == 0)
            x,y = np.where(background)
            empty_mask = {}
            empty_mask['segmentation'] = np.zeros_like(self.label_image, dtype=bool)
            empty_mask['indexes'] = [x,y]

            # choose random index for point_coords
            idx = np.random.randint(len(x))
            empty_mask['point_coords'] = [[x[idx], y[idx]]]

            # add pointer to image
            if self.image is not None:
                empty_mask['image'] = self.image
                
            self.mask_list.append(empty_mask)

    def make_3d_label_image(self):
        num_masks = len(self.mask_list)
        mask_shape = [num_masks, *self.mask_list[0]['segmentation'].shape]
        self.label_image = np.zeros(mask_shape, dtype=np.uint16)
        for i, mask in enumerate(self.mask_list):
            self.label_image[i] = mask['segmentation']*(i+1)

    def make_2d_labels(self, type="min"):
        self.make_3d_label_image()

        if type == "min":
            # Create a masked array where zeros are masked
            masked_label_image = np.ma.masked_equal(self.label_image, 0)
            # Perform the min projection on the masked array
            _2d_labels = np.ma.min(masked_label_image, axis=0).filled(0)
        else:
            _2d_labels = np.max(self.label_image, axis=0)

        return _2d_labels
    
    