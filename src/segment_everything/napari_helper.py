import napari
from segment_everything.stacked_labels import StackedLabels
from  napari_segment_everything import segment_everything

def stacked_labels_to_napari(stacked_labels):
    viewer = napari.Viewer()
    segment_everything_widget=segment_everything.NapariSegmentEverything(viewer)
    viewer.window.add_dock_widget(segment_everything_widget)
    segment_everything_widget.load_project(stacked_labels.image, stacked_labels.mask_list)

def to_napari(image, mask_list):
    viewer = napari.Viewer()
    segment_everything_widget=segment_everything.NapariSegmentEverything(viewer)
    viewer.window.add_dock_widget(segment_everything_widget)
    segment_everything_widget.load_project(image, mask_list)