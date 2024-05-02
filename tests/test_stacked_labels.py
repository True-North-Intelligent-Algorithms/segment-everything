import raster_geometry as rg
import numpy as np
import matplotlib.pyplot as plt
from segment_everything.stacked_labels import StackedLabels

def test_make_stacked_labels_from_2d():
    nx, ny = (500, 500)
    circle1 = rg.circle((nx, ny), 50, (0.3, 0.3) )
    circle2 = rg.circle((nx, ny), 50, (0.7, 0.7) )
    circle3 = rg.circle((nx, ny), 50, (0.7, 0.3) )

    binary = circle1 + circle2+ circle3

    #plt.imshow(binary)
    #plt.show()
    stacked_labels = StackedLabels.from_2d_label_image(binary, None)

    assert len(stacked_labels.mask_list) == 3

def test_make_stacked_labels_from_masks():
    nx, ny = (500, 500)
    circle1 = rg.circle((nx, ny), 50, (0.3, 0.3) )
    circle2 = rg.circle((nx, ny), 50, (0.7, 0.7) )
    circle3 = rg.circle((nx, ny), 50, (0.7, 0.3) )
    
    stacked_labels = StackedLabels()

    stacked_labels.add_segmentation(circle1)
    stacked_labels.add_segmentation(circle2)
    stacked_labels.add_segmentation(circle3)

    assert len(stacked_labels.mask_list) == 3

    for i in range(3):
        print(stacked_labels.mask_list[i]['point_coords'])

    for i in stacked_labels.mask_list[2]['indexes']:
        print('x,y',i)