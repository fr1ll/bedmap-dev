# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_utils.ipynb.

# %% auto 0
__all__ = [
    "FILE_NAME",
    "get_version",
    "round_floats",
    "date_to_seconds",
    "round_date",
    "timestamp",
    "datestring_to_date",
    "get_path",
    "write_json",
    "read_json",
    "clean_filename",
]

# %% ../../nbs/01_utils.ipynb 2
import datetime
import gzip
import json
import os
from urllib.parse import unquote

from dateutil.parser import parse as parse_date

# %% ../../nbs/01_utils.ipynb 3
FILE_NAME = "filename"  # Filename name key


# %% ../../nbs/01_utils.ipynb 4
def get_version():
    """
    Return the version of bedmap installed
    Hardcoded for now
    """
    # return pkg_resources.get_distribution("bedmap").version
    return "0.0.1"


# %% ../../nbs/01_utils.ipynb 6
def round_floats(obj, digits=5):
    """Return 2D array obj with rounded float precision"""
    return [[round(float(j), digits) for j in i] for i in obj]


# %% ../../nbs/01_utils.ipynb 8
def date_to_seconds(date):
    """
    Given a datetime object return an integer representation for that datetime
    """
    if isinstance(date, datetime.datetime):
        return (date - datetime.datetime.today()).total_seconds()
    else:
        return -float("inf")


def round_date(date, unit):
    """
    Return `date` truncated to the temporal unit specified in `units`
    """
    if not isinstance(date, datetime.datetime):
        return "no_date"
    formatted = date.strftime("%d %B %Y -- %X")
    if unit in set("seconds", "minutes", "hours"):
        date = formatted.split("--")[1].strip()
        if unit == "seconds":
            date = date
        elif unit == "minutes":
            date = ":".join(date.split(":")[:-1]) + ":00"
        elif unit == "hours":
            date = date.split(":")[0] + ":00:00"
    elif unit in set("days", "months", "years", "decades", "centuries"):
        date = formatted.split("--")[0].strip()
        if unit == "days":
            date = date
        elif unit == "months":
            date = " ".join(date.split()[1:])
        elif unit == "years":
            date = date.split()[-1]
        elif unit == "decades":
            date = str(int(date.split()[-1]) // 10) + "0"
        elif unit == "centuries":
            date = str(int(date.split()[-1]) // 100) + "00"
    return date


# %% ../../nbs/01_utils.ipynb 9
def timestamp():
    """Return a string for printing the current time"""
    return str(datetime.datetime.now()) + ":"


# %% ../../nbs/01_utils.ipynb 10
def datestring_to_date(datestring):
    """
    Given a string representing a date return a datetime object
    """
    try:
        return parse_date(str(datestring), fuzzy=True, default=datetime.datetime(9999, 1, 1))
    except Exception:
        print(timestamp(), f"Could not parse datestring {datestring}")
        return datestring


# %% ../../nbs/01_utils.ipynb 12
def get_path(*args, **kwargs):
    """Return the path to a JSON file with conditional gz extension

    Args:
        sub_dir (str)
        filename (str)
        out_dir (str)
        add_hash (Optional[bool])
        plot_id (Optional[str]): Required if add_hash is True
        gzip (Optional[bool])

    """
    sub_dir, filename = args
    if sub_dir:
        out_dir = os.path.join(kwargs["out_dir"], sub_dir)
    else:
        out_dir = kwargs["out_dir"]

    if kwargs.get("add_hash", True):
        filename += f"-{kwargs['plot_id']}"

    path = os.path.join(out_dir, filename + ".json")
    if kwargs.get("gzip", False):
        path += ".gz"

    return path


# %% ../../nbs/01_utils.ipynb 13
def write_json(path, obj, **kwargs):
    """Write json object `obj` to disk and return the path to that file

    Args:
        path (str)
        obj (json serializable)
        gzip (Optional[bool]): Default = False
        encoding (str): Required if gzip = True
    """
    out_dir, filename = os.path.split(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if kwargs.get("gzip", False):
        with gzip.GzipFile(path, "w") as out:
            out.write(json.dumps(obj, indent=4).encode(kwargs["encoding"]))
        return path
    else:
        with open(path, "w") as out:
            json.dump(obj, out, indent=4)
        return path


def read_json(path, **kwargs):
    """Read and return the json object written by the current process at `path`

    Args:
        path (str)
        gzip (Optional[bool]): Default = False
        encoding (str): Required if gzip = True

    """
    if kwargs.get("gzip", False):
        with gzip.GzipFile(path, "r") as f:
            return json.loads(f.read().decode(kwargs["encoding"]))
    with open(path) as f:
        return json.load(f)


# %% ../../nbs/01_utils.ipynb 16
def clean_filename(s, **kwargs):
    """Given a string that points to a filename, return a clean filename

    Args:
        s (str): filename path

    Returns:
        s (str): clean file name

    Notes:
        kwargs is not used at all

    """
    s = unquote(os.path.basename(s))
    invalid_chars = '<>:;,"/\\|?*[]'
    for i in invalid_chars:
        s = s.replace(i, "")
    return s
