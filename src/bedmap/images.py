# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03_images.ipynb.

# %% auto 0
__all__ = [
    "PILLoadTruncated",
    "load_image",
    "resize_to_max_side",
    "resize_to_height",
    "autocontrast",
    "create_atlases_and_thumbs",
    "get_image_paths",
    "ValidImage",
    "find_duplicates",
    "ImageFactoryBase",
    "ImageFactory",
]

# %% ../../nbs/03_images.ipynb 2
import copy
import os
import random
from abc import ABC, abstractmethod
from glob import glob

import numpy as np
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from .metadata import get_metadata_list
from .utils import FILE_NAME, clean_filename, timestamp

# handle truncated images in PIL (managed by Pillow)
PILLoadTruncated = ImageFile.LOAD_TRUNCATED_IMAGES

# imports when switching to PIL-only resizing
from pathlib import Path


# %% ../../nbs/03_images.ipynb 4
def load_image(image_path: str, format: str = "RGB") -> Image:
    """load an image and convert to desired format"""
    return Image.open(image_path).convert(format)


# %% ../../nbs/03_images.ipynb 7
def resize_to_max_side(img: Image, n: int = 128):
    """
    resize to a maximum side length
    """
    w, h = img.size
    if w > h:
        # preserve ratio but ensure height is >=1 pixel
        size = (n, max(1, int(n * h / w)))
    else:
        # preserve ratio but ensure width is >=1 pixel
        size = (max(1, int(n * w / h)), n)
    return img.resize(size, reducing_gap=2.0)


# %% ../../nbs/03_images.ipynb 8
def resize_to_height(img: Image, height: int = 128):
    """
    resize to an exact height
    """
    w, h = img.size
    if (w / h * height) < 1:
        resizedwidth = 1
    else:
        resizedwidth = int(w / h * height)
    size = (resizedwidth, height)
    return img.resize(size, reducing_gap=2.0)


# %% ../../nbs/03_images.ipynb 10
def autocontrast(img: Image) -> Image:
    """autocontrast lifted from keras library
    added lightness normalization"""
    x = np.asarray(img, dtype=float)
    mean_before = x.mean()
    ## autocontrast from Keras
    x = x - np.min(x)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    ## return to average lightness of input image
    mean_shift = x.mean() - mean_before
    x = np.clip(x - mean_shift, 0, 255)
    return Image.fromarray(x.astype("uint8"))


# %% ../../nbs/03_images.ipynb 14
def create_atlases_and_thumbs(imageEngine, plot_id, use_cache: bool = False):
    """create folder with atlases in data dir"""

    print(timestamp(), "Copying images to output directory")

    # create directories
    atlas_dir = Path(imageEngine.out_dir) / "atlases" / str(plot_id)
    atlas_dir.mkdir(exist_ok=True, parents=True)

    thumbs_dir = Path(imageEngine.out_dir) / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)

    orig_dir = Path(imageEngine.out_dir) / "originals"
    orig_dir.mkdir(exist_ok=True)

    # initialize some atlas values
    n_atlases, x, y = 0, 0, 0
    positions = []
    atlas_size = (imageEngine.atlas_size, imageEngine.atlas_size)

    for img in tqdm(imageEngine, total=imageEngine.count):
        # copy thumbnail
        thumb = resize_to_max_side(img.original, n=imageEngine.lod_cell_height)
        thumb = autocontrast(thumb)
        thumb_w, thumb_h = thumb.width, thumb.height
        thumb.save(thumbs_dir / img.unique_name)

        # copy resized original
        fullsize_path = orig_dir / img.unique_name
        if use_cache and fullsize_path.exists():
            pass
        else:
            fullsize = resize_to_height(img.original, height=600)
            fullsize.save(orig_dir / img.unique_name)

        # create atlas
        cell = resize_to_height(img.original, height=imageEngine.cell_size)
        cell = autocontrast(cell)

        if (x + cell.width) > atlas_size[0]:  # end of a row
            y += cell.height
            x = 0
        if (y + cell.height) > atlas_size[0]:  # end of first column
            atlas.save(atlas_dir / f"atlas-{n_atlases}.jpg")
            n_atlases += 1
            x, y = 0, 0  # start a new atlas
        if x == 0 and y == 0:
            atlas = Image.new(mode="RGB", size=atlas_size)
        atlas.paste(cell, (x, y))

        # store in dict
        positions.append({"idx": n_atlases, "x": x, "y": y, "w": thumb_w, "h": thumb_h})
        x += cell.width

    if not (x == 0 and y == 0):  # if last atlas wasn't already written
        atlas.save(atlas_dir / f"atlas-{n_atlases}.jpg")
    return atlas_dir.as_posix(), positions


# %% ../../nbs/03_images.ipynb 15
def get_image_paths(images: str, out_dir: str) -> list[str]:
    """Called once to provide a list of image paths.

    args:
        images (str): directory location of images.
        out_dir (str): output directory for downloaded IIIF files.

    returns:
        image_paths list(str): list of image paths.

    TODO: Remove out_dir parameter since it was ony used for IIIF files
    and the IIIF functionality is removed.

    """

    image_paths = None

    # allow images to be input as list, i.e. from tables input
    if isinstance(images, list):
        image_paths = images

    # handle case where images flag points to a glob of images
    if not image_paths:
        image_paths = glob(images, recursive=True)

    # handle case user provided no images
    if not image_paths:
        raise FileNotFoundError("Error: No input images were found. Please check your --images glob")

    return image_paths


# %% ../../nbs/03_images.ipynb 17
class ValidImage:
    """
    TODO: Convert to dataclass
    TODO: Rename
    """

    def __init__(self, img_path: str, metadata: dict | None = None) -> "ValidImage":
        self.path = img_path
        self._original = None
        self._filename = None
        self._unique_name = None
        self.metadata = metadata if metadata else {}

    @property
    def original(self):
        if self._original is None:
            self._original = load_image(self.path)
        return self._original

    @property
    def filename(self):
        if self._filename is None:
            self._filename = clean_filename(self.path)
        return self._filename

    @property
    def unique_name(self):
        """Save as name when copying image."""
        if self._unique_name is None:
            self._unique_name = self.filename
        return self._unique_name

    def valid(self, lod_cell_height: int, oblong_ratio: int | float) -> tuple[bool, str]:
        """Validate that image can be opened and loaded correctly.

        Args:
            lod_cell_height (int):
            oblong_ratio (int|float): atlas_size/cell_size ratio

        Returns:
            Tuple[pass,msg]:
                pass (bool): True if passed validation
                msg (str): Reason why validation failed
        """
        w, h = self.original.size
        # remove images with 0 height or width when resized to lod height
        if (h == 0) or (w == 0):
            return False, f"Skipping {self.path} because it contains 0 height or width"
        # remove images that have 0 height or width when resized
        try:
            _ = resize_to_height(self.original, height=lod_cell_height)
        except ValueError:
            return False, f"Skipping {self.path} because it contains 0 height or width when resized"
        except OSError:
            return False, f"Skipping {self.path} because it could not be resized"
        # remove images that are too wide for the atlas
        if (w / h) > (oblong_ratio):
            return False, f"Skipping {self.path} because its dimensions are oblong"

        return True, ""


# %% ../../nbs/03_images.ipynb 19
def find_duplicates(maybe_dups: list) -> set:
    """return any duplicates from a list"""
    seen = set()
    return {x for x in maybe_dups if x in seen or seen.add(x)}


# %% ../../nbs/03_images.ipynb 20
class ImageFactoryBase(ABC):
    """Class encapsulates functionality required to access images,
    including compiling metadata.

    Factory is responsible :
        - Compiling image files and their metadata
        - Filtering and validating images
        - Naming image output names
        - Providing property values

    Image factory needs to be able to provide an Image instance
        - The image instance needs to be have it's metadata (if applicable)

    Notes:
        Class and their children are required to provide the following properties:
            out_dir
            shuffle
            atlas_size
            cell_size
            lod_cell_height
    """

    # Required property values
    DEFAULT_OPTIONS = {
        "shuffle": False,  # (Optional[bool], default = False): Shuffle image order
        "atlas_size": 2048,  # (int, default = 2048)
        "cell_size": 32,  # (int, default = 32)
        "lod_cell_height": 128,  # (int, default = 128)
    }

    def __init__(self, out_dir: str) -> None:
        """Initialize ImageEngine instance

        Args:
            out_dir (str): output directory of data

        """
        # Required initialized properties
        self.out_dir = out_dir
        self.count = 0  # Total number of images
        self.meta_headers = []  # Headers in metadata
        self.metadata = []  # List of metadata

        # Required property values
        self.shuffle = self.DEFAULT_OPTIONS["shuffle"]
        self.atlas_size = self.DEFAULT_OPTIONS["atlas_size"]
        self.cell_size = self.DEFAULT_OPTIONS["cell_size"]
        self.lod_cell_height = self.DEFAULT_OPTIONS["lod_cell_height"]

    @abstractmethod
    def __iter__(self):
        # Yield a ValidImage instance
        pass

    @abstractmethod
    def __getitem__(self, index):
        # Return index's ValidImage instance
        pass


class ImageFactory(ImageFactoryBase):
    # Default internal values
    DEFAULT_OPTIONS = ImageFactoryBase.DEFAULT_OPTIONS.copy()
    DEFAULT_OPTIONS.update(
        {
            "seed": "",  # (int): Seed for random generator
            "max_images": False,  # (Union[False,int]): Maximum number of images
            "validate": True,  # Validate Images
        }
    )

    def __init__(self, img_path, out_dir, meta_dir, options={}) -> None:
        super().__init__(out_dir)
        self.img_path = img_path
        self.meta_dir = meta_dir
        self.filenames = []
        self.image_paths = []

        # Load options
        for option, default in self.DEFAULT_OPTIONS.items():
            setattr(self, option, options.get(option, default))

        self.filter_images()

    def __iter__(self):
        for img in self.stream_images(self.image_paths, self.metadata):
            yield img

    def __getitem__(self, index):
        if index < len(self.image_paths):
            if self.metadata:
                meta = self.metadata[index]
            else:
                meta = None

            return ValidImage(self.image_paths[index], meta)

    def filter_images(self):
        """Main method for filtering images given user metadata (if provided)

        -Validate image:
            Loading (done by stream_images and Images)
            Size
            resizing
            oblong

        -Compare against metadata

        Args:
            images (str): Directory location of images.
            out_dir (str): Output directory.
            shuffle (Optional[bool], default = False): Shuffle image order
            seed (int): Seed for random generator
            max_images (Union[bool,int]): Maximum number of images
            atlas_size (int, default = 2048)
            cell_size (int, default = 32)
            lod_cell_height (int, default = 128)
            meta_dir (str): Directory of image metadata

        Returns:
            images (list[str])
            metadata (list[dict])

        Notes:
            Assumes 'filename' is provided in metadata
            Convoluted compiling of metadata
            Should All Validation should belong to Image class?
            Need to split function
        """
        # validate that input image names are unique
        image_paths = get_image_paths(images=self.img_path, out_dir=self.out_dir)
        image_names = list(map(clean_filename, image_paths))
        duplicates = set([x for x in image_names if image_names.count(x) > 1])

        if duplicates:
            raise Exception(
                """Image filenames should be unique, but the following 
                filenames are duplicated\n{}""".format("\n".join(duplicates))
            )

        # optionally shuffle the image_paths
        if self.shuffle:
            print(timestamp(), "Shuffling input images")
            random.Random(self.seed).shuffle(image_paths)
        else:
            image_paths = sorted(image_paths)

        # Optionally limit the number of images in image_paths
        if self.max_images:
            image_paths = image_paths[: self.max_images]

        # process and filter the images
        filtered_image_paths = {}
        oblong_ratio = self.atlas_size / self.cell_size

        print(timestamp(), "Validating input images")
        for img in tqdm(self.stream_images(image_paths, []), total=len(image_paths)):
            if self.validate is True:
                valid, msg = img.valid(lod_cell_height=self.lod_cell_height, oblong_ratio=oblong_ratio)
                if valid is True:
                    filtered_image_paths[img.path] = img.filename
                else:
                    print(timestamp(), msg)
            else:
                filtered_image_paths[img.path] = img.filename

        # if there are no remaining images, throw an error
        if len(filtered_image_paths) == 0:
            raise Exception("No images were found! Please check your input image glob.")

        # handle the case user provided no metadata
        if not self.meta_dir:
            print(timestamp(), "No metadata directory was provided.")
            self.image_paths = list(filtered_image_paths.keys())
            self.count = len(self.image_paths)
            self.filenames = list(filtered_image_paths.values())
            return

        # handle user metadata: retain only records with image and metadata
        metaList, self.meta_headers = get_metadata_list(meta_dir=self.meta_dir)
        metaDict = {clean_filename(i.get(FILE_NAME, "")): i for i in metaList}
        meta_bn = set(metaDict.keys())
        img_bn = set(filtered_image_paths.values())

        # identify images with metadata and those without metadata
        meta_present = img_bn.intersection(meta_bn)
        meta_missing = list(img_bn - meta_bn)

        # notify the user of images that are missing metadata
        if meta_missing:
            print(
                timestamp(),
                " ! Some images are missing metadata:\n  -",
                "\n  - ".join(meta_missing[:10]),
            )
            if len(meta_missing) > 10:
                print(timestamp(), " ...", len(meta_missing) - 10, "more")

            if os.path.exists(self.out_dir) is False:
                os.makedirs(self.out_dir)

            missing_dir = os.path.join(self.out_dir, "missing-metadata.txt")
            with open(missing_dir, "w") as out:
                out.write("\n".join(meta_missing))

        if not meta_present:
            raise Exception(
                f"""No image has matching metadata. Check if '{FILE_NAME}' key was provided in metadata files"""
            )

        # get the sorted lists of images and metadata
        images = []
        metadata = []
        for path, fileName in filtered_image_paths.items():
            if fileName in meta_present:
                images.append(path)
                metadata.append(copy.deepcopy(metaDict[fileName]))
                self.filenames.append(fileName)

        self.image_paths = images
        self.metadata = metadata
        self.count = len(self.image_paths)

    @staticmethod
    def stream_images(image_paths: list[str], metadata: list[dict] | None = None) -> "ValidImage":
        """Read in all images from args[0], a list of image paths

        Args:
            image_paths (list[str]): list of image locations
            metadata (Optional[list[dist]]): metadata for each image

        Returns:
            yields ValidImage instance

        Notes:
            image is matched to metadata by index location
                Matching by key would be better
        """
        for idx, imgPath in enumerate(image_paths):
            try:
                meta = None
                if metadata and metadata[idx]:
                    meta = metadata[idx]
                yield ValidImage(imgPath, meta)
            except Exception as exc:
                print(timestamp(), "Image", imgPath, "could not be processed --", exc)
