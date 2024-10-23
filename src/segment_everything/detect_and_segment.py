#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from segment_everything.weights_helper import create_sam_model
from segment_everything.stacked_labels import StackedLabels
from segment_everything.vendored.mobilesamv2 import SamPredictor as SamPredictorV2
from segment_everything.weights_helper import get_device
from typing import Any, Generator, List
import torch
import os
from segment_anything.utils.amg import calculate_stability_score
import gc

current_dir = os.path.dirname(__file__)

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(
        len(args[0]) % batch_size != 0
    )
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def segment_from_stacked_labels(stacked_labels, model_type, device=None):
    """ given stacked labels and a model re-segment all masks by calling sam on each bounding box

        this function is useful when we have stacked labels with bounding boxes only (or other non-optimal masks)
        and we want to resegment the masks using sam

    Args:
        stacked_labels (StackedLabels): input stacked labels 
        model (SAM model): sam model 
        device (string): device type

    Returns:
        StackedLabels: new stacked labels with resegmented labels 
    """
    if device is None:
        device = get_device()
    model = create_sam_model(model_type, device)
    bbox_array = stacked_labels.get_bbox_np()
    sam_masks = segment_from_bbox(stacked_labels.image, bbox_array, model, device)

    return StackedLabels(sam_masks, stacked_labels.image)

def segment_from_bbox(img, bounding_boxes, model, device):
    """
    Segments everything given the bounding boxes of the objects and the mobileSAMv2 prediction model.
    Code from mobileSAMv2
    """
    predictor = SamPredictorV2(model)
    predictor.set_image(img)
    
    input_boxes = predictor.transform.apply_boxes(
        bounding_boxes, predictor.original_size
    )  # Does this need to be transformed?
    if device == "cuda":
        input_boxes = torch.from_numpy(input_boxes).cuda()
    elif device == "cpu":
        input_boxes = torch.from_numpy(input_boxes)
    sam_mask = []

    predicted_ious = []
    stability_scores = []

    image_embedding = predictor.features
    image_embedding = torch.repeat_interleave(image_embedding, 200, dim=0)

    prompt_embedding = model.prompt_encoder.get_dense_pe()
    prompt_embedding = torch.repeat_interleave(prompt_embedding, 200, dim=0)

    for (boxes,) in batch_iterator(200, input_boxes):
        with torch.no_grad():
            image_embedding = image_embedding[0 : boxes.shape[0], :, :, :]
            prompt_embedding = prompt_embedding[0 : boxes.shape[0], :, :, :]
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
            low_res_masks, pred_ious = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=prompt_embedding,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                simple_type=True,
            )
            low_res_masks = predictor.model.postprocess_masks(
                low_res_masks, predictor.input_size, predictor.original_size
            )
            model.threshold_offset = 1
            stability_score = (
                calculate_stability_score(
                    low_res_masks,
                    model.mask_threshold,
                    model.threshold_offset,
                )
                .cpu()
                .numpy()
            )
            sam_mask_pre = (low_res_masks > model.mask_threshold) * 1.0
            sam_mask.append(sam_mask_pre.squeeze(1))
            predicted_ious.extend(pred_ious.cpu().numpy().flatten().tolist())
            stability_scores.extend(stability_score.flatten().tolist())

    sam_mask = torch.cat(sam_mask)
    # predicted_ious = pred_ious.cpu().numpy()
    cpu_segmentations = sam_mask.cpu().numpy()
    del sam_mask

    gc.collect()
    torch.cuda.empty_cache()

    curr_anns = []
    for idx in range(len(cpu_segmentations)):
        ann = {
            "segmentation": cpu_segmentations[idx],
            "area": sum(sum(cpu_segmentations[idx])),
            "predicted_iou": predicted_ious[idx],
            "stability_score": stability_scores[idx],
            "prompt_bbox": bounding_boxes[idx],
        }
        if (
            cpu_segmentations[idx].max() < 1
        ):  # this means that bboxes won't always == segmentations
            continue
        curr_anns.append(ann)
    return curr_anns
