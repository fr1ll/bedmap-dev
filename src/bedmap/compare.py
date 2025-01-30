# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/02_compare.ipynb.

# %% auto 0
__all__ = [
    "project_root",
    "clip_plt_dir",
    "baseline_dir",
    "temp_dir",
    "baseline_proj_root",
    "MANIFEST1",
    "MANIFEST2",
    "COMPARE_FILES",
    "log_output",
    "clean_diff_output",
    "delete_temp",
    "manifest_replace_write",
    "copy_file",
    "fix_expected_diff",
    "try_read_text",
    "useful_text_diff",
    "compare_named_files",
    "comFile",
]

# %% ../../nbs/02_compare.ipynb 4
import difflib
import filecmp
import json
from pathlib import Path
from shutil import rmtree

from . import bedmap, utils

# %% ../../nbs/02_compare.ipynb 5
# calling project_root instead of previous name of basedir to avoid confusion with baseline_dir
project_root = bedmap.get_bedmap_root()

clip_plt_dir = project_root / "tests/smithsonian_butterflies_10/output_test_temp"
baseline_dir = project_root / "tests/butterflies_baseline"
temp_dir = project_root / "tests/butterflies_baseline_temp"

baseline_proj_root = "/home/carlo/source/bedmap"

# %% ../../nbs/02_compare.ipynb 7
MANIFEST1 = "data/manifests/manifest-test_diff.json"
MANIFEST2 = "data/manifest.json"

# Files to track and compare
COMPARE_FILES = [
    "data/atlases/test_diff/atlas_positions.json",
    "data/hotspots/hotspot-test_diff.json",
    "data/imagelists/imagelist-test_diff.json",
    # Layouts
    "data/layouts/rasterfairy-test_diff.json",
    "data/layouts/categorical-labels-test_diff.json",
    "data/layouts/categorical-test_diff.json",
    "data/layouts/timeline-test_diff.json",
    "data/layouts/timeline-labels-test_diff.json",
    "data/layouts/umap-test_diff.json",
    # Manifest
    MANIFEST1,
    MANIFEST2,
    # Metadata (per image)
    "data/metadata/file/329c2b4d5-8137-414b-b98e-ff04907e8ea6.jpg.json",
    "data/metadata/file/0dae7d86-9c14-11ed-a00f-a37ce258aeb3.jpg.json",
    "data/metadata/file/9fea3150-a3d4-11ed-aeea-e36f1256f233.jpg.json",
    "data/metadata/file/3fee89f9b-ba5c-4f2e-8532-6e390e2cf0c9.jpg.json",
    "data/metadata/file/31fd87d81-7ff5-4e50-b718-0d4a259d47c1.jpg.json",
    "data/metadata/file/36829498c-0eda-4c84-b89d-c893c75cfa68.jpg.json",
    "data/metadata/file/354747d21-638c-4e23-a655-c0bb4de18941.jpg.json",
    "data/metadata/file/376dd3835-ea28-48ab-b7e8-c7c929a77a01.jpg.json",
    "data/metadata/file/30aeb051d-ee0d-4c5b-8a85-a8da7baef5fd.jpg.json",
    "data/metadata/file/3c3407493-e0d9-43fe-a2a2-7c43395c90c5.jpg.json",
    "data/metadata/file/329a4c094-8536-4396-be70-3d9b5d0744d9.jpg.json",
    "data/metadata/file/386168016-7276-4b02-b713-5f36ac2ef452.jpg.json",
    # Metadata
    "data/metadata/filters/filters.json",
    "data/metadata/options/Beautiful.json",
    "data/metadata/options/Tiny.json",
    "data/metadata/options/Spots.json",
    "data/metadata/options/Colorful.json",
    "data/metadata/options/Broken.json",
    "data/metadata/options/Gray.json",
    "data/metadata/options/Ugly.json",
    "data/metadata/options/Moth.json",
    "data/metadata/dates.json",
]


# %% ../../nbs/02_compare.ipynb 9
def log_output(txt: str, space: int | None = 0) -> None:
    # Hook for logging results
    if isinstance(txt, str):
        print(f'{" " * space}{txt}')
    else:
        print("".join(txt))


# %% ../../nbs/02_compare.ipynb 10
def clean_diff_output(msg: list[str]) -> str:
    """Clean differ.compare() output to only show difference"""
    cleanMsg = ""
    for line in msg:
        if line.startswith("?") or line.startswith("-") or line.startswith("+"):
            cleanMsg += line
    return cleanMsg


def delete_temp():
    """Delete temporary directory"""
    if temp_dir.exists():
        rmtree(temp_dir)


# %% ../../nbs/02_compare.ipynb 11
def manifest_replace_write(manifest_filename):
    """replace dates and filenames in baseline manifest files"""
    new_m = (clip_plt_dir / manifest_filename).read_text()
    old_m = (temp_dir / manifest_filename).read_text()

    # get dates
    new_date = json.loads(new_m)["creation_date"]
    old_date = json.loads(old_m)["creation_date"]

    # replace dates and root so they should match new manifest
    old_m = old_m.replace(old_date, new_date)
    old_m = old_m.replace(baseline_proj_root, project_root.as_posix())

    # write manifest after replacements
    (temp_dir / manifest_filename).write_text(old_m)


# %% ../../nbs/02_compare.ipynb 12
def copy_file():
    """copy baseline directory to temporary directory"""
    delete_temp()
    utils.copytree_agnostic(baseline_dir, temp_dir)


def fix_expected_diff():
    """Update pix_plot manifest to match bedmap manifest

    Updates only:
        Creation date
        Directory references
    """
    # Replace date and directory references
    manifest_replace_write(MANIFEST1)
    manifest_replace_write(MANIFEST2)


# %% ../../nbs/02_compare.ipynb 14
def try_read_text(filename: Path) -> list[str] | None:
    """Check if file can be read as text"""
    try:
        with filename.open() as f:
            txtlines = f.readlines()
        return txtlines
    except (UnicodeDecodeError, IsADirectoryError):
        return None


# %% ../../nbs/02_compare.ipynb 15
def useful_text_diff(file1: str, file2: str) -> tuple[bool | None, str]:
    """This function compares two files to return text difference, if they are textfiles.

    Args:
        file1 (str): path for file 1
        file2 (str): path for file 2

    Returns:
        tuple(bool, str):
            bool: true if files have a text difference
            str: feedback if file difference
    """
    try:
        if filecmp.cmp(file1, file2) is False:
            file1_lines = try_read_text(file1)
            file2_lines = try_read_text(file2)
            if file1_lines is not None and file2_lines is not None:
                differ = difflib.Differ()
                diff = list(differ.compare(file1_lines, file2_lines))
                diff = clean_diff_output(diff)
                if diff == "":
                    return False, ""
                return True, diff
            else:
                return None, "Could not read files as text"
        else:
            return False, ""
    except FileNotFoundError:
        return None, "Could not open files"


# %% ../../nbs/02_compare.ipynb 16
def compare_named_files():
    """Function loops named files in the COMPARE_FILES
    list to compare the files form the legacy output and
    the new output.
    """
    for j, file in enumerate(COMPARE_FILES):
        log_output(f"\n#{j} Comparing {file}")
        file_clip = clip_plt_dir / file
        file_pix = baseline_dir / file
        chk, txt = useful_text_diff(file_clip, file_pix)
        if chk is True:
            log_output("- Fail", 2)
            log_output(txt, 4)
        else:
            log_output("+ Identical", 2)


# %% ../../nbs/02_compare.ipynb 18
def comFile():
    """Compare two directories

    Call out missing files and folders and files with
    different content.
    """
    try:
        # Create temporary copy of base line
        copy_file()

        # Edit temp copy and fix expected difference
        fix_expected_diff()

        fail = False
        log_output("Comparing Files and folders")
        fileComp = filecmp.dircmp(clip_plt_dir, temp_dir)

        # Create a queue
        currs = [fileComp]
        while currs:
            curr = currs.pop()

            try:
                # Check for different files
                if curr.diff_files:
                    loc = curr.right.replace(str(temp_dir), "")
                    # This difference can relate to mtime alone -- not important in itself
                    # log_output(f'Different at {loc}:\n    {curr.diff_files}\n',2)
                    for asset in curr.diff_files:
                        has_useful_diff, msg = useful_text_diff(Path(curr.right) / asset, Path(curr.left) / asset)
                        if has_useful_diff is True:
                            log_output(f"{asset} Difference", 2)
                            log_output(msg, 2)
                        else:
                            log_output(f"{asset} Difference", 2)
                            log_output("Unrecognized difference", 2)
                    fail = True

                # Update queue
                for k, v in curr.subdirs.items():
                    currs.append(v)

                # Check if baseline_dir is missing files/folders
                for asset in curr.left_list:
                    if asset not in curr.right_list:
                        fail = True
                        log_output(f"Pix-plot Missing: {asset}", 2)

                # Check if clip_plt_dir is missing files/folders
                for asset in curr.right_list:
                    if asset not in curr.left_list:
                        fail = True
                        log_output(f"bedmap Missing: {asset}", 2)

            except FileNotFoundError as e:
                msg = f"{str(e)} \n   Check directories path for clip_plt_dir and baseline_dir!"
                log_output(msg, 2)

    except Exception as e:
        print(e)
        fail = True

    finally:
        delete_temp()

    if fail is False:
        log_output("No difference found!")

    return fail


# %% ../../nbs/02_compare.ipynb 19
if __name__ == "__main__":
    comFile()
