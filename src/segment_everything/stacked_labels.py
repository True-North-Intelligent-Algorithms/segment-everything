import numpy as np

from skimage.measure import label, regionprops
from skimage import color
from skimage.measure import regionprops

class StackedLabels:
    """
    A class to manage and manipulate a list of masks 

    Attributes:
    -----------
    image : numpy.ndarray
        The input image. 
    label_image : numpy.ndarray
        The label image where each unique integer represents a different object.
    mask_list : list
        A list of masks, each represented as a dictionary containing various attributes such as 
        segmentation, bounding box, point coordinates, area, etc.  The segmentation attribute contains the 
        binary representation of the mask. 

    Methods:
    --------
    __init__(self, mask_list=None, image=None, label_image=None)
        Initializes the StackedLabels object with optional mask list, image, and label image.
    
    create_mask_from_segmentation(segmentation, image=None)
        Static method to create a mask dictionary from a segmentation array and optionally an image.
    
    create_mask_from_yolo_bbox(bbox_yolo, image)
        Static method to create a mask dictionary from a YOLO bounding box and an image.
    
    get_bbox(segmentation)
        Static method to calculate the bounding box of a given segmentation array.
    
    read_yolo_txt(file_path)
        Static method to read YOLO format results from a text file.
    
    from_2d_label_image(cls, label_image, image, relabel=True)
        Class method to create a StackedLabels object from a 2D label image and an image.
    
    from_yolo_results(cls, results, image, relabel=True)
        Class method to create a StackedLabels object from YOLO results and an image.
    
    add_segmentation(self, segmentation)
        Adds a new segmentation to the mask list.
    
    add_background_results(self, num_background_results=1)
        Adds empty masks for background regions to the mask list.
    
    make_3d_label_image(self)
        Constructs a 3D label image from the mask list.
    
    make_2d_labels(self, type="min")
        Creates a 2D label image by performing a min or max projection of the 3D label image.
    
    get_bbox_np(self)
        Returns a numpy array of bounding boxes for all masks in the mask list.
    """

    def __init__(self, mask_list=None, image = None, label_image=None):
        """
        Initializes the StackedLabels object.

        Parameters:
        -----------
        mask_list : list, optional
            A list of mask dictionaries to initialize with. Each dictionary represents a mask and contains
            various attributes such as segmentation, bounding box, point coordinates, area, etc.
            Defaults to an empty list if not provided.
        image : numpy.ndarray, optional
            The input image. If provided and is a 2D array, it is converted to a 3-channel image by stacking
            the 2D array along the last axis. This is done to ensure the image has three channels, which is
            often required for further processing by SAM models.
        label_image : numpy.ndarray, optional
            The label image to initialize with. The label image is a 2D array where each unique integer
            represents a different object or region. This can be useful for creating masks and further
            segmenting the image.
        """
        self.image = image

        if image is not None and image.ndim == 2:
            self.image = np.stack([self.image, self.image, self.image], axis=-1)

        self._3d_labels = None
        self._2d_labels = None
        self.label_image = label_image  # legacy, for backward compatibility (deprecate soon)

        if mask_list is None:
                self.mask_list = []
        else:
            self.mask_list = mask_list

            for mask in self.mask_list:
                mask['keep'] = True

        self.mask_list = sorted(self.mask_list, key=lambda x: x['area'], reverse=True)

    @staticmethod
    def create_mask_from_segmentation(segmentation, image=None):
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
    def create_mask_from_xywhn_bbox(bbox_yolo, image):
        """
        Creates a mask from a YOLO bounding box.

        -----------
        bbox_yolo : list or tuple
            The YOLO bounding box, typically in the format [x_center, y_center, width, height].
            The coordinates and dimensions are normalized to the range [0, 1].
        image : numpy.ndarray
            The image on which the bounding box is defined. The shape of the image is used to convert
            the normalized coordinates and dimensions of the bounding box to pixel values.

        Returns:
        --------
        dict
            A dictionary representing the mask created from the YOLO bounding box. The dictionary contains
            various attributes such as the segmentation mask, bounding box, point coordinates, area, and
            other relevant information.

        Notes:
        ------
        - The method converts the YOLO bounding box coordinates from normalized values to pixel values
        based on the dimensions of the input image.
        - The segmentation mask is created by setting the pixels within the bounding box to True.
        """
        # need to convert bbox to pixels
        x, y, w, h = bbox_yolo
        w = abs(w)
        h = abs(h)
        x_min, y_min = (x-w/2, y-h/2)
        x_max, y_max = x+w/2, y+h/2
        x_min = x_min * image.shape[1]
        x_max = x_max * image.shape[1]
        y_min = y_min * image.shape[0]
        y_max = y_max * image.shape[0]

        return StackedLabels.create_mask_from_xyxy_bbox([x_min, y_min, x_max, y_max], image)
     
    @staticmethod 
    def create_mask_from_xyxy_bbox(bbox_yolo, image):
        """
        Creates a mask from a YOLO xyxy bounding box.

        Parameters:
        -----------
        bbox_yolo : list or tuple
            The YOLO bounding box 
        image : numpy.ndarray
            The image on which the bounding box is defined. 

        Returns:
        --------
        dict
            A dictionary representing the mask created from the YOLO bounding box. The dictionary contains
            various attributes such as the segmentation mask, bounding box, point coordinates, area, and
            other relevant information.

        Notes:
        ------
        - The method converts the YOLO bounding box coordinates from normalized values to pixel values
        based on the dimensions of the input image.
        - The segmentation mask is created by setting the pixels within the bounding box to True.
        """
        # need to convert bbox to pixels
        x_min, y_min, x_max, y_max = bbox_yolo
        segmentation = np.zeros(image.shape[:2], dtype=bool)
        segmentation[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = True
        if segmentation.sum() == 0:
            print('empty')

        return StackedLabels.create_mask_from_segmentation(segmentation, image)
    
    @staticmethod    
    def get_bbox(segmentation):
        """
        Computes the bounding box of a segmentation mask.

        Parameters:
        -----------
        segmentation : numpy.ndarray
            A 2D binary array representing the segmentation mask. Pixels belonging to the segmented
            object are marked with True (or 1), and the background pixels are marked with False (or 0).

        Returns:
        --------
        list
            A list of four integers representing the bounding box of the segmented object in the format
            [x_min, y_min, x_max, y_max]. These coordinates define the smallest rectangle that can
            contain all the True pixels in the segmentation mask.
        """
        y, x = np.where(segmentation > 0)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return [x_min, y_min, x_max, y_max]
    
    @staticmethod
    def read_yolo_txt(file_path):
        """
        Reads a YOLO format text file and extracts bounding box information.

        Parameters:
        -----------
        file_path : str
            The path to the YOLO format text file. Each line in the file represents one bounding box and
            contains the class ID followed by the normalized coordinates and dimensions of the bounding
            box (x_center, y_center, width, height).

        Returns:
        --------
        list
            A list of dictionaries, each containing the following keys:
            - 'class_id': int, the class ID of the object.
            - 'bbox': list of floats, the bounding box coordinates and dimensions in the format 
            [x_center, y_center, width, height], with values normalized to the range [0, 1].

        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        results = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            results.append({'class_id': class_id, 'bbox': bbox})

        return results

    @classmethod
    def from_2d_label_image(cls, label_image, image, relabel=True):
        """
        Creates an instance of StackedLabels from a 2D label image.

        Parameters:
        -----------
        label_image : numpy.ndarray
            A 2D array where each unique integer represents a different object or region. This label image
            is used to create segmentation masks for each unique region.
        image : numpy.ndarray
            The input image corresponding to the label image. 
            to each mask.
        relabel : bool, optional
            If True, the label image is relabeled to ensure that the labels are consecutive integers starting
            from 1. This can be useful if the label image has gaps, repeat labels or non-sequential labels. Default is True.

        Returns:
        --------
        StackedLabels
            An instance of the StackedLabels class, initialized with masks created from the label image and
            the input image.
        """
        if (relabel):
            label_image = label(label_image)

        mask_list = []
        for region in regionprops(label_image):
            segmentation = np.zeros_like(label_image, dtype=bool)
            segmentation[label_image == region.label] = True
            mask = cls.create_mask_from_segmentation(segmentation, image)
            mask_list.append(mask)

        return cls(mask_list, image, label_image)
    
    @classmethod
    def from_yolo_results(cls, bboxes, classes, image):
        """
        Creates an instance of StackedLabels from YOLO detection results.

        Parameters:
        -----------
        results : list
            A list of dictionaries, each containing the following keys:
            - 'class_id': int, the class ID of the object.
            - 'bbox': list of floats, the bounding box coordinates and dimensions in the format 
            [x_center, y_center, width, height], with values normalized to the range [0, 1].
        image : numpy.ndarray
            The input image corresponding to the YOLO detection results. This image is used to create
            segmentation masks from the bounding boxes.

        Returns:
        --------
        StackedLabels
            An instance of the StackedLabels class, initialized with masks created from the YOLO detection
            results and the input image.
        """
        mask_list = []
        for i in range(bboxes.shape[0]):
            #class_id = classes[i]
            bbox = bboxes[i,:]

            mask = cls.create_mask_from_xyxy_bbox(bbox, image)
            mask_list.append(mask)

        return cls(mask_list, image)

    @classmethod
    def from_yolo_dictionary(cls, results, image, format='xyxy'):
        """
        Creates an instance of StackedLabels from YOLO format dictionary that is often generated when labelling
        (ie with napari or other bounding box drawing tool)

        Parameters:
        -----------
        results : list
            A list of dictionaries, each containing the following keys:
            - 'class_id': int, the class ID of the object.
            - 'bbox': list of floats, the bounding box coordinates and dimensions in the format 
            [x_center, y_center, width, height], with values normalized to the range [0, 1].
        image : numpy.ndarray
            The input image corresponding to the YOLO detection results. This image is used to create
            segmentation masks from the bounding boxes.

        Returns:
        --------
        StackedLabels
            An instance of the StackedLabels class, initialized with masks created from the YOLO detection
            results and the input image.
        """
        mask_list = []
        for result in results:
            class_id = result['class_id']
            bbox = result['bbox']

            if format == 'xyxy':
                mask = cls.create_mask_from_xyxy_bbox(bbox, image)
            else:
                mask = cls.create_mask_from_xywhn_bbox(bbox, image)
            mask_list.append(mask)

        return cls(mask_list, image)

    def add_segmentation(self, segmentation):
        """
        Adds a new dictionary to the mask list.

        Parameters:
        -----------
        segmentation : numpy.ndarray
            A boolean array where True indicates the presence of an object.
        """
        stacked_label = self.create_mask_from_segmentation(segmentation)
        self.mask_list.append(stacked_label)

    def add_background_results(self, num_background_results=1):
        """
        Adds empty masks for background regions to the mask list.  this is useful if training a model and we 
        want to avoid false positives. Thus we add empty masks to represent the background (or objects which should 
        not be detected).

        Parameters:
        -----------
        num_background_results : int, optional
            The number of background results to add. Defaults to 1.
        """

        for i in range(num_background_results):
            # Use 2D label image if available, else fallback to zeros
            bg_shape = self.get_2d_labels().shape if self._2d_labels is not None else self.mask_list[0]['segmentation'].shape
            background = np.zeros(bg_shape, dtype=bool)
            x, y = np.where(background == 0)
            empty_mask = {}
            empty_mask['segmentation'] = np.zeros_like(background, dtype=bool)
            empty_mask['indexes'] = [x, y]
            idx = np.random.randint(len(x))
            empty_mask['point_coords'] = [[x[idx], y[idx]]]
            if self.image is not None:
                empty_mask['image'] = self.image
            self.mask_list.append(empty_mask)

    def make_3d_labels(self):
        """
        Make and return a 3D label image from the mask list, each layer contains a single mask.
        Returns:
            numpy.ndarray: 3D label image (num_masks, H, W)
        """
        num_masks = len(self.mask_list)
        mask_shape = [num_masks, *self.mask_list[0]['segmentation'].shape]
        labels_3d = np.zeros(mask_shape, dtype=np.uint16)
        for i, mask in enumerate(self.mask_list):
            if mask['keep'] == True:
                labels_3d[i] = mask['segmentation']*(i+1)
        return labels_3d

    def make_2d_labels(self, type="min"):
        """
        Make and return a 2D label image by projecting the 3D label image.
        Args:
            type (str): 'min' or 'max' projection.
        Returns:
            numpy.ndarray: 2D label image.
        """
        labels_3d = self.get_3d_labels()
        if type == "min":
            masked_label_image = np.ma.masked_equal(labels_3d, 0)
            labels_2d = np.ma.min(masked_label_image, axis=0).filled(0)
        else:
            labels_2d = np.max(labels_3d, axis=0)
        return labels_2d

    def get_3d_labels(self):
        """
        Returns the 3D label image, computing it if necessary.
        """
        if self._3d_labels is None:
            self._3d_labels = self.make_3d_labels()
        return self._3d_labels

    def get_2d_labels(self, type="min"):
        """
        Returns the 2D label image, computing it if necessary.
        """
        if self._2d_labels is None:
            self._2d_labels = self.make_2d_labels(type=type)
        return self._2d_labels
    
    def get_bbox_np(self):
        """
        Get the bounding boxes for all masks in the mask list as a numpy array"
        
        Returns:
        --------
        numpy.ndarray
            A numpy array of bounding boxes for all masks in the mask list.
        """
        bboxes = []
        for mask in self.mask_list:
            bboxes.append(mask['prompt_bbox'])
        return np.array(bboxes)

    def sort_largest_to_smallest(self):
        """
        Sorts the mask list from largest to smallest area.
        """ 
        self.mask_list = sorted(self.mask_list, key=lambda x: x['area'], reverse=True)
    

    def add_properties_to_label_image(self):
        """
        Add properties to the label image such as area, circularity, solidity, mean intensity, 10th percentile intensity,
        
        Returns:
        --------
        
        """

        self.sort_largest_to_smallest()
        sorted_results = self.mask_list

        hsv_image = color.rgb2hsv(self.image)
        # switch to this? https://forum.image.sc/t/looking-for-a-faster-version-of-rgb2hsv/95214/12

        hue = 255 * hsv_image[:, :, 0]
        saturation = 255 * hsv_image[:, :, 1]
        intensity = 255 * hsv_image[:, :, 2]

        for enum, result in enumerate(sorted_results):
            segmentation = result["segmentation"]
            coords = np.where(segmentation)
            regions = regionprops(segmentation.astype("uint8"))

            # calculate circularity
            result["circularity"] = (
                4 * np.pi * regions[0].area / (regions[0].perimeter ** 2)
            )
            # for small pixelated objects, circularity can be > 1 so we cap it
            if result["circularity"] > 1:
                result["circularity"] = 1

            result["solidity"] = regions[0].solidity
            intensity_pixels = intensity[coords]
            result["mean_intensity"] = np.mean(intensity_pixels)
            result["10th_percentile_intensity"] = np.percentile(
                intensity_pixels, 10
            )
            
            hue_pixels = hue[coords]

            # Convert hue to radians for circular mean calculation
            hue_radians = np.deg2rad(hue_pixels)

            # Calculate the circular mean
            mean_x = np.mean(np.cos(hue_radians))
            mean_y = np.mean(np.sin(hue_radians))
            mean_angle = np.arctan2(mean_y, mean_x)

            # Convert back to degrees and normalize to [0, 360]
            mean_hue = np.rad2deg(mean_angle) % 360
            result["mean_hue"] = mean_hue
            
            saturation_pixels = saturation[coords]
            result["mean_saturation"] = np.mean(saturation_pixels)

            result['keep'] = True
            result["label_num"] = enum + 1


    def filter_labels_3d_multi(self, stat_limits): 
        """
        Filter the labels in the label image based on the stat_limits dictionary.
        The stat_limits dictionary contains the minimum and maximum values for each statistic that a label must meet

        Parameters:
        -----------
        stat_limits : dict
            A dictionary containing the minimum and maximum values for each statistic that a label must meet.
            The dictionary should have the following format:
            {
                "area": {"min": 100, "max": 1000},
                "circularity": {"min": 0.5, "max": 1.0},
                "solidity": {"min": 0.5, "max": 1.0},
                "mean_intensity": {"min": 100, "max": 200},
                "10th_percentile_intensity": {"min": 50, "max": 150},
                "mean_hue": {"min": 0, "max": 100},
                "mean_saturation": {"min": 0, "max": 100},
        """
        sorted_results = self.mask_list
        label_image = self.get_3d_labels()
        
        for enum, result in enumerate(sorted_results):
            # Check if the result should be kept based on the limits in stat_limits
            keep = all(
                stat_limits[stat]["min"] <= result[stat] <= stat_limits[stat]["max"]
                for stat in stat_limits
            )
           
            if keep:
                if result["keep"] == True:
                    continue
                result["keep"] = True
                coords = np.where(result["segmentation"])
                temp = label_image[enum, :, :]
                temp[coords] = enum + 1
            else:
                if result["keep"] == False:
                    continue
                result["keep"] = False
                coords = np.where(result["segmentation"])
                temp = label_image[enum, :, :]
                temp[coords] = 0

   
    def filter_labels_hue_inverse(self, hue_min, hue_max):
        """
            Inverse filter hue.  This is needed because hue wraps around.  For example redish hue goes from
            about 330 degrees to 30.  Thus to keep only red objects we need to inverse filter (keep everything outside of hue min and max)
        """
        sorted_results = self.mask_list
        label_image = self.get_3d_labels()
        
        for enum, result in enumerate(sorted_results):
            hue = result["mean_hue"]

            if result["keep"]==True:
                keep=False
                if hue < hue_min:
                    keep = True
                elif hue > hue_max:
                    keep = True
                else:
                    keep = False

                if keep:
                    if result["keep"] == True:
                        continue
                    result["keep"] = True
                    coords = np.where(result["segmentation"])
                    temp = label_image[enum, :, :]
                    temp[coords] = enum + 1
                else:
                    if result["keep"] == False:
                        continue
                    result["keep"] = False
                    coords = np.where(result["segmentation"])
                    temp = label_image[enum, :, :]
                    temp[coords] = 0
 
        

                
        
            
            