"""
Module containing various utility functions and classes.
"""
from argparse import Namespace
from dataclasses import dataclass, asdict
from io import IOBase
from itertools import groupby
from operator import attrgetter, itemgetter
from os import environ, X_OK, pathsep, access
from pathlib import Path
from sys import stdout
from typing import IO, Generator, Union, Iterable, Callable
from urllib.parse import urlencode
from urllib.request import urlopen, Request


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass  # (kw_only=True) https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class Config:
    """
    Config parent class that can conveniently set attributes from CLI args
    """

    @classmethod
    def from_args(cls, args: Namespace):
        """
        Sets attributes of the class from a Namespace object (e.g. from argparse)

        Parameters
        ----------
        args : :class:`argparse.Namespace`
            :class:`argparse.Namespace` object containing attributes to set

        Returns
        -------
        cls
            Class instance with attributes set from args

        """
        return cls(**{f: getattr(args, f) for f in asdict(cls) if hasattr(args, f)})

# Functions ------------------------------------------------------------------------------------------------------------
def find_executable_binaries(*programs: str) -> Generator[Path, None, None]:
    """
    Check if programs are installed and executable

    :param programs: List of programs to check
    :return: Generator of Path objects for each program found
    """
    programs = set(programs)
    for path in map(Path, environ["PATH"].split(pathsep)):
        if path.is_dir():
            for binary in path.iterdir():
                if binary.name in programs and access(binary, X_OK):
                    yield binary


def is_non_empty_file(file: Union[str, Path], min_size: int = 1) -> bool:
    """
    Checks if a file exists, is a file, and is non-empty (optionally above a minimum size).

    :param file: Path to the file to check.
    :param min_size: Minimum size of the file in bytes.
    :return: True if the file exists, is a file, and is non-empty, False otherwise.
    """
    if not isinstance(file, Path):
        file = Path(file)
    if file.exists() and file.is_file():
        return file.stat().st_size >= min_size
    else:
        return False


def write_to_file_or_directory(path: Union[str, Path, IO], mode: str = 'at') -> Union[Path, IO]:
    """
    Writes to a file or creates a directory based on the provided path.

    If the path is '-' or 'stdout', it returns stdout.
    If the path has a suffix, it's treated as a file and opened for appending.
    If the path has no suffix, it's treated as a directory and created if it doesn't exist.

    :param path: The path to the file or directory.
    :param mode: The mode to open the file in if it's a file.
    :return: A file handle (IO) if a file is specified, or a Path object if a directory is specified.
    """
    if isinstance(path, IOBase):
        return path
    if path in {'-', 'stdout'}:  # If the path is '-', return stdout
        return stdout
    if not isinstance(path, Path):  # Coerce to Path object
        path = Path(path)
    if path.suffix:  # If the path has an extension, it's probably a file
        # NB: We can't use is_file or is_dir because it may not exist yet, `open()` will create or append
        return open(path, mode)  # Open the file
    else:
        path.mkdir(exist_ok=True, parents=True)  # Create the directory if it doesn't exist
    return path


def download(url: Union[str, Request], dest: Union[str, Path] = None, data = None, encode_data: bool = False) -> Union[Path, bytes, None]:
    """
    Downloads a file from a URL to a destination or returns the data as bytes.

    :param url: URL to download from.
    :param dest: Destination path to save the file to. If None, returns the data as bytes.
    :return: Path to the downloaded file or the data as bytes.
    :raises ValueError: If no data is written to the destination file.
    """
    if data and encode_data:
        data = urlencode(data).encode('utf-8')
    if not isinstance(url, Request):
        url = Request(url, headers={'User-Agent': 'Mozilla/5.0'}, data=data)
    response = urlopen(url)
    if dest:
        with open(dest, mode='wb') as handle:
            handle.write(response.read())
        if (dest := Path(dest)).stat().st_size == 0:
            dest.unlink()
            raise ValueError('No data written')
        return dest
    else:
        return response.read()


def grouper(iterable: Iterable, key: Union[str, int, Callable]):
    """Shortcut for sorting and grouping"""
    if isinstance(key, str): key = attrgetter(key)
    elif isinstance(key, int): key = itemgetter(key)
    yield from groupby(sorted(iterable, key=key), key=key)


def bold(text: str):
    """
    Makes text bold in the terminal.

    :param text: Text to make bold.
    :return: Bold text.
    """
    return f"\033[1m{text}\033[0m"
