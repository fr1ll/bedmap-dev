# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04_embeddings.ipynb.

# %% auto 0
__all__ = ["DEVICE", "timm_embed_model", "timm_transform_embed", "get_timm_embeds"]

# %% ../../nbs/04_embeddings.ipynb 3
from .utils import timestamp, clean_filename

from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
import timm

# %% ../../nbs/04_embeddings.ipynb 4
# global variables to help inference performance based on device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if DEVICE.type == "cpu":
    TORCH_DTYPE = torch.float32
elif torch.cuda.is_bf16_supported():
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float16

print(f"Device for inference: {DEVICE}; dtype for inference: {TORCH_DTYPE}")


# %% ../../nbs/04_embeddings.ipynb 6
def timm_embed_model(model_name: str):
    """
    Load model and image transform to create embeddings
    Reference: https://huggingface.co/docs/timm/main/en/feature_extraction#pooled

    input:          model name as found in timm documentation
    return tuple:   pre-trained embedding model,
                    transform function to prep images for inference
    """

    m = timm.create_model(model_name, pretrained=True, num_classes=0)

    # Reference on transform: https://huggingface.co/docs/timm/main/en/feature_extraction#pooled
    t = timm.data.create_transform(**timm.data.resolve_data_config(m.pretrained_cfg))

    m = m.eval().to(device=DEVICE, dtype=TORCH_DTYPE)
    m = torch.jit.optimize_for_inference(torch.jit.script(m))
    return m, t


# %% ../../nbs/04_embeddings.ipynb 7
def timm_transform_embed(img, model, transform, device, dtype) -> np.ndarray:
    """
    apply transform to image and run inference on it to generate an embedding

    input:      img: Pillow image or similar
                model: Torch model
                transform: Torch image transformation pipeline to match how model was trained
    returns: embedding vector as 1D numpy array
    """
    img = transform(img).to(device, dtype).unsqueeze(0)
    emb = model(img)

    return emb.detach().cpu().float().numpy().squeeze()


# %% ../../nbs/04_embeddings.ipynb 8
def get_timm_embeds(imageEngine, model_name: str, data_dir: Path, **kwargs):
    """
    Create embedding vectors for input images using a pre-trained model from timm
    """
    # for now, the output directory is still called "inception" though it is generic
    vector_dir = data_dir / "image-vectors" / "inception"
    vector_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(kwargs["seed"])

    print(timestamp(), f"Creating embeddings using {model_name}")
    embeds = []
    embed_paths = []

    model, transform = timm_embed_model(model_name)

    for img in tqdm(imageEngine, total=imageEngine.count):
        embed_path = vector_dir / (clean_filename(img.path) + ".npy")
        if embed_path.exists() and kwargs["use_cache"]:
            emb = np.load(embed_path)
        else:
            # create embedding for one image
            emb = timm_transform_embed(img.original, model, transform, DEVICE, TORCH_DTYPE)
            np.save(embed_path, emb)
        embeds.append(emb)
        embed_paths.append(embed_path.as_posix())
    return np.array(embeds), embed_paths
