#!/usr/bin/env python

# In[1]:


import gc
import shutil
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path

import daft
import numpy as np
import timm
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# In[2]:


BATCH_SIZE = 32
MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
IMAGE_GLOB = None
IMAGES_FOLDER = "./tmp-test-images"

TEST_DATASET = "kvriza8/microscopy_images"
NUM_TEST_IMAGES = 500

nice_models = [
"mobilenetv3_large_100",
"vit_small_patch14_reg4_dinov2.lvd142m",
"vit_base_patch14_reg4_dinov2.lvd142m",
"vit_large_patch14_reg4_dinov2.lvd142m",
"aimv2_large_patch14_224.apple_pt_dist"
]


# with vit_base_patch14 and torch dataloader:
#
# num_images | batch_size | optimize | time |
# -----------|------------|----------|------|
# 500        |         32 | False    | 10:07
# 200        |         32 | False    | 04:50
# 2000       |         32 | False    | 41:00
# 50         |         32 | Static   | 01:14
# 50         |         32 | Dynamic  | 01:14
# 500        |         32 | Static   | 09:33
# 2000       | 32 (fixed) | Static   | 36:22
# 2000       | 16 (fixed) | Static   | 36:32
# 2000       |  4 (fixed) | Static   | 39:17
# 2000       | 128 (fixd) | Static   | OOM
# 2000lrg    |  16 (fixd) | Static   | -

# In[3]:


def dl_hf_images(dataset_name: str = "kvriza8/microscopy_images",
                 dir: Path = None,
                 max_images: int = 50,
                 overwrite: bool = True) -> None:

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    if overwrite:
        shutil.rmtree(dir, ignore_errors=True)
        dir.mkdir(parents=True, exist_ok=True)

    for i, img_row in enumerate(tqdm(iter(dataset), total=max_images)):
        if i >= max_images:
            break
        img = img_row["image"]
        img.save(dir / f"{i}.png")

    del dataset
    gc.collect()

    return None


# In[4]:


tmp_path = Path(IMAGES_FOLDER)
dl_hf_images(dir=tmp_path, max_images=NUM_TEST_IMAGES)


# In[5]:


@dataclass
class Embedder:
    model_name: str
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: torch.dtype = field(init=False)
    model: torch.nn.Module = field(init=False)
    transform: callable = field(init=False)

    def __post_init__(self):
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model.to(self.device, memory_format=torch.channels_last)
        self.model.eval()
        self.model = torch.compile(self.model, dynamic=True, mode="reduce-overhead")
        # Resolve config removes unneeded fields before create_transform
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg)

    @torch.inference_mode()
    def embed(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of pre-transformed images, compute pooled embeddings.
        The batch is moved to the proper device (with channels_last format) and processed in inference mode.
        """
        batch_imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        if self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=self.dtype):
                return self.model(batch_imgs)
        else:
            # autocast can be comically slow for some CPU setups (PyTorch issue #118499)
            return self.model(batch_imgs)


# In[6]:


@daft.udf(return_dtype=daft.DataType.python())
class TransformImageCol:
    """run timm embedder on an image column"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedder = Embedder(self.model_name)

    def __call__(self, batch_images) -> list:
        return [self.embedder.transform(Image.fromarray(im)) for im in batch_images.to_pylist()]


class ImageListIteratorAsDict(Dataset):
    def __init__(self, filelist: list[Path], transform: callable):
        self.filelist = filelist
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx: int):
        image = Image.open(self.filelist[idx]).convert("RGB")
        if self.transform:
            return {"image_transformed": self.transform(image)}
        # return as dict for easy comparison vs. daft
        else:
            return {"image": [image]}


def get_file_list(source: str | list[str]) -> list[Path]:
    """
    Given a folder path, a glob pattern, or a filelist, return a list of image file paths.
    If source is a directory, a glob is run using the provided pattern.
    If source is a string containing a wildcard, glob is applied.
    Otherwise, if it's a list, it is returned directly.
    """
    if isinstance(source, list):
        return [Path(s) for s in source]
    elif Path(source).is_dir():
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        return list(chain.from_iterable([Path(source).glob(p) for p in patterns]))
    elif isinstance(source, str) and '*' in source:
        return [Path(p) for p in glob(source)]
    else:
        return [source]

# In[7]:


imglob = tmp_path.as_posix() +"/*.png"
images_df = daft.from_glob_path(imglob).with_column_renamed("path", "path_full_img")
images_df = images_df.with_column("image", daft.col("path_full_img"
                                 ).url.download().image.decode(
                                     mode="RGB", on_error="null")
                                 )
images_df = images_df.where(images_df["image"].not_null())

TransformImForModel = TransformImageCol.with_init_args(model_name=MODEL_NAME)

images_df = images_df.with_column("image_transformed", TransformImForModel(daft.col("image"))
                                  ).exclude("image", "num_rows")

images_df.show(1)


# In[8]:


def compute_embeddings(model_name: str,
                       batch_size: int = BATCH_SIZE,
                       images_df: None | daft.DataFrame = None,
                       images_folder: None | Path = None) -> list[np.ndarray]:
    """
    Given a model name and a filelist (list of image paths), this function computes and returns a list
    of embeddings (one per image). The function instantiates an Embedder, builds a dataset and dataloader,
    and processes images in batches.
    """
    embedder = Embedder(model_name=model_name)

    if images_df is not None:
        dataset = images_df.to_torch_iter_dataset()
    elif images_folder:
        # use vanilla torch dataloader
        image_list = get_file_list(images_folder)
        dataset = ImageListIteratorAsDict(image_list, embedder.transform)


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    embeddings = []
    for i, batch_images in enumerate(tqdm(dataloader, unit_scale=BATCH_SIZE)):
        emb = embedder.embed(batch_images["image_transformed"]).cpu().numpy()

        if i == 0:
            print(f"Shape of embedding for one batch: {emb.shape}")
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    return embeddings


# In[9]:
USE_DAFT: bool = True

if USE_DAFT:
    embeddings = compute_embeddings(MODEL_NAME, BATCH_SIZE, images_df=images_df)
else:
    # use vanilla torch dataset
    embeddings = compute_embeddings(MODEL_NAME, BATCH_SIZE, images_folder=IMAGES_FOLDER)
