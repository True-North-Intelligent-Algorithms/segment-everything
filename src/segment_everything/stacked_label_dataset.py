from torch.utils.data import Dataset
import random
import numpy as np

class StackedLabelDataset(Dataset):
    """
    This class is used to create a dataset with stacked labels.  For an m by n image each label image is also
    m by n array and contains only one label.   This way overlapping labels can be handled.  
    """
    def __init__(self, data, processor):
        """ initializes the StackedLabelDataset

        Args:
            data (list): A list of dictionary objects.  Each dictionary object contains an image and a label collection
            processor (_type_): _description_
        """
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """ returns an item from the dataset

        Args:
            idx (int): the index of the item to return

        Returns:
            dict: a dictionary containing the image and label collection
        """
        result = self.data[idx]
        image = result['image']

        # if number dims 2
        if len(image.shape) == 2:
            # add fake channels
            image = np.stack([image, image, image], axis=-1)

        segmentation = result['segmentation']

        if 'prompt_bbox' in result:
            x_min, y_min, x_max, y_max = result['prompt_bbox']
            H, W = segmentation.shape
            x_min = max(0, x_min - np.random.randint(0, 10))
            x_max = min(W, x_max + np.random.randint(0, 10))
            y_min = max(0, y_min - np.random.randint(0, 10))
            y_max = min(H, y_max + np.random.randint(0, 10))
            box_prompt = [x_min, y_min, x_max, y_max]
            box_prompt = [float(x) for x in box_prompt]
            box_prompt = [[box_prompt]]
        else:
            box_prompt = None

        if self.random_index == False:
            point_prompt = result['point_coords']
        else:
            y, x = result['indexes']
            idx = np.random.randint(len(x))
            point_prompt = [[x[idx], y[idx]]]

        '''
            if np.sum(segmentation) == 0:
                print('empty')
            else:
                print('mask')

            print('centroid is', result['point_coords'])
            print('random prompt', point_prompt)
            print()
        '''
        '''
        if len(x) > 0:
            idx = np.random.randint(len(x))
            prompt = [[x[idx], y[idx]]]
        else:    
            prompt = result['point_coords']
        '''        
        '''
        number_results = len(results)

        # random id for label
        results_id = random.randint(0, 4*number_results)

        # if the label is from the original labels
        if results_id < number_results:
            result = results[results_id]
            segmentation = result['segmentation']
            y, x = np.where(segmentation)
            idx = np.random.randint(len(x))
            prompt = [[x[idx], y[idx]]]
        else:
            all_labels = item['binary']
            not_labeled = np.logical_not(all_labels)
            y, x = np.where(not_labeled)
            idx = np.random.randint(len(x))
            prompt = [[x[idx], y[idx]]]
            segmentation = np.zeros_like(all_labels, dtype=bool)
        #print('random prompt', prompt)
        #print()
        '''
        #print('image shape is',image.shape)
        #print('prompt is',prompt)

        inputs = self.processor(image, input_points=point_prompt, input_boxes = box_prompt, return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs['ground_truth'] = segmentation.astype('float32')

        # prompt = 
        return inputs