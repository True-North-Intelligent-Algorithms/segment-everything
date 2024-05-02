from segment_everything.weights_helper import create_mobile_sam_model

def test_download_mobile_sam_weights():
    model = create_mobile_sam_model()
    assert model is not None
