# import distance map
from typing import Optional
from pathlib import Path
import toolz as tz
from napari.utils import progress
import urllib.request
import warnings
import gdown
import os
import torch

current_dir = os.path.dirname(__file__)

from segment_everything.vendored.mobilesamv2 import (
    sam_model_registry,
)

WEIGHTS_URLS = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "efficientvit_l2": "https://drive.google.com/uc?id=10Emd1k9obcXZZALiqlW8FLIYZTfLs-xu/l2.pt",
    "ObjectAwareModel_Cell_FT": "https://drive.google.com/uc?id=1efZ40ti87O346dJW5lp7inCZ84N_nugS/ObjectAwareModel_Cell_FT.pt",
    "ObjectAwareModel": "https://drive.google.com/uc?id=1_vb_0SHBUnQhtg5SEE24kOog9_5Qpk5Z/ObjectAwareModel.pt",
    "PromptGuidedDecoder": "local:vendored/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt",
    "ObjectAwareModelHuggingFace": "https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/ObjectAwareModel.pt",
}

@tz.curry
def _report_hook(
    block_num: int,
    block_size: int,
    total_size: int,
    pbr: "progress" = None,
) -> None:
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size
    downloaded_mb = downloaded / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    increment = int(percent) - pbr.n
    if increment > 1:  # faster than increment at every iteration
        pbr.update(increment)
    print(
        f"Download progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_size_mb:.1f} MB)",
        end="\r",
    )


def download_weights(weight_url: str, weight_path: "Path"):
    print(f"Downloading {weight_url} to {weight_path} ...")
    pbr = progress(total=100)
    try:
        if weight_url.startswith("https://drive.google.com/"):
            google_weight_url = "/".join(weight_url.split("/")[0:-1])

            gdown.download(google_weight_url, str(weight_path))
        else:
            urllib.request.urlretrieve(
                weight_url, weight_path, reporthook=_report_hook(pbr=pbr)
            )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        urllib.error.ContentTooShortError,
    ) as e:
        warnings.warn(f"Error downloading {weight_url}: {e}")
        return None
    else:
        print("\rDownload complete.                            ")
    pbr.close()


def get_weights_path(model_type: str) -> Optional[Path]:
    """Returns the path to the weight of a given model architecture."""
    weight_url = WEIGHTS_URLS[model_type]

    if weight_url.startswith("local:"):
        weight_path = os.path.join(current_dir, weight_url.split(":")[-1])
        return weight_path

    cache_dir = Path.home() / ".cache/segment_everything"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / weight_url.split("/")[-1]

    # Download the weights if they don't exist
    if not weight_path.exists():
        download_weights(weight_url, weight_path)

    return weight_path

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def create_mobile_sam_model():
    weights_path_VIT = get_weights_path("efficientvit_l2")
    weights_path_prompt_guided = get_weights_path("PromptGuidedDecoder")
    prompt_guided_decoder = sam_model_registry["PromptGuidedDecoder"](weights_path_prompt_guided)
    mobilesamv2 = sam_model_registry["vit_h"]()
    
    mobilesamv2.prompt_encoder = prompt_guided_decoder["PromtEncoder"]
    mobilesamv2.mask_decoder = prompt_guided_decoder["MaskDecoder"]
    mobilesamv2.image_encoder = sam_model_registry["efficientvit_l2"](weights_path_VIT)

    return mobilesamv2

def create_sam_model(model_type, device=None):
    if model_type == "MobileSamV2":
        model = create_mobile_sam_model()
    else:
        model = sam_model_registry[model_type](get_weights_path(model_type))

    if device is None:
        device = get_device()
    model.to(device)

    return model

    
