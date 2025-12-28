from segment_everything.mask_detectors.mobilesam import mobilesam_detector


def test_download_mobile_sam_weights():
    detector = mobilesam_detector(model_type="MobileSamV2", device="cpu")
    # Ensure the underlying model object was created
    assert detector.model is not None
