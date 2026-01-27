"""
Module containing various utility functions and classes.
"""
from json import loads as json_loads
from argparse import Namespace
from dataclasses import dataclass, fields
from io import BytesIO
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from itertools import groupby
from operator import attrgetter, itemgetter
from os import environ, X_OK, pathsep, access
from pathlib import Path
from typing import Generator, Union, Iterable, Callable, Any
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from shutil import copyfileobj, which
from time import time
from sys import stderr


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True)  # (kw_only=True) https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class Config:
    """
    Config parent class that can conveniently set attributes from CLI args
    """
    @classmethod
    def from_args(cls, args: Namespace):
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)})


class LiteralFile:
    """Class for handling real files"""
    _MIN_SIZE = 1

    def __init__(self, path: Path, size):
        self._path = path
        self._size = size
        self._name_parts = path.name.split('.')

    @classmethod
    def from_path(cls, path: Union[str, Path], return_bool: bool = False):
        if not isinstance(path, Path): path = Path(path)
        if not path.exists():
            if return_bool: return False
            raise FileNotFoundError(f'File {path} does not exist')
        if not path.is_file():
            if return_bool: return False
            raise FileNotFoundError(f'File {path} is not a file')
        size = path.stat().st_size
        if size < cls._MIN_SIZE:
            if return_bool: return False
            raise FileNotFoundError(f'File {path} is too small')
        return cls(path, size)

    @property
    def path(self): return self._path
    @property
    def size(self): return self._size
    @property
    def name(self): return self._path.name
    @property
    def stem(self): return self._name_parts[0]
    @property
    def suffix(self): return self._name_parts[1]


class GitRepo:
    """
    A context-aware class to represent a git repository.
    Handles temporary directories automatically via the 'with' statement.

    Examples:
        >>> with GitRepo('owner', 'repo') as repo:
        ...     print(repo.local_path)
    """
    _BASE_URL = 'https://api.github.com'
    __slots__ = ('owner', 'repo', 'branch', '_api_url', '_meta_cache', '_temp_dir_obj', 'local_path')
    def __init__(self, owner: str, repo: str, branch: str = 'main'):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self._api_url = f'{self._BASE_URL}/repos/{owner}/{repo}'
        # Internal state
        self._meta_cache = None
        self._temp_dir_obj = None  # Holds the TemporaryDirectory object
        self.local_path: Path | None = None

    def __repr__(self): return f"<GitRepo {self.owner}/{self.repo} ({self.branch})>"

    @property
    def metadata(self) -> dict:
        """
        Lazy-loaded property for repository metadata.
        Only downloads from API once per instance.
        """
        if self._meta_cache is None: self._meta_cache = json_loads(download(self._api_url))
        return self._meta_cache

    def clone(self) -> Path:
        """
        Downloads and extracts the repo to a temporary directory.
        Returns the Path to the actual source code (inside the extracted folder).
        """
        # If already cloned, just return the path
        if self.local_path and self.local_path.exists(): return self.local_path
        zip_url = f'{self._api_url}/zipball/{self.branch}'
        # Stream download to avoid memory spikes if repo is large
        # (Modified logic to handle streaming would go here, keeping your simple logic for now)
        data = download(zip_url)
        # Create the temp directory explicitly
        self._temp_dir_obj = TemporaryDirectory()
        base_temp_path = Path(self._temp_dir_obj.name)
        with BytesIO(data) as zip_buffer:
            with ZipFile(zip_buffer, 'r') as zfile:
                zfile.extractall(base_temp_path)
                # GitHub zips wrap everything in a single top-level folder
                # We want self.local_path to point INSIDE that folder
                root_folder = zfile.namelist()[0].split('/')[0]
        self.local_path = base_temp_path / root_folder
        return self.local_path

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.clone()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and cleanup temporary directory."""
        if self._temp_dir_obj:
            self._temp_dir_obj.cleanup()
            self._temp_dir_obj = None
            self.local_path = None


class ProgressBar:
    """
    A CLI progress bar that wraps an iterable and calls a function for each item.

    This class acts as an iterator, yielding results from the callable while
    printing a dynamic progress bar to stderr.

    Args:
        iterable (Iterable): An iterable of items to process.
        callable (Callable): A function to call for each item from the iterable.
        desc (str): A description to display before the progress bar.
        bar_length (int): The character length of the progress bar.
        bar_character (str): The character used to fill the progress bar.
        silence (bool): Silence the bar
    """
    __slots__ = ('iterable', 'total', 'callable', 'desc', 'bar_length', 'bar_character', 'silence', '_iterator',
                 'start_time', '_processed_count')
    def __init__(self, iterable: Iterable, callable: Callable[[Any], Any], desc: str = "Processing items",
                 bar_length: int = 40, bar_character: str = '█', silence: bool = False):
        # Eagerly consume the iterable to get a total count for the progress bar.
        assert len(bar_character) == 1, "Bar character must be a single character"
        self.items = list(iterable)
        self.total = len(self.items)
        self.callable = callable
        self.desc = desc
        self.bar_length = bar_length
        self.bar_character = bar_character
        self.silence = silence
        self._iterator: Iterable = None
        self.start_time: float = None
        self._processed_count: int = 0

    def __len__(self): return self.total

    def __iter__(self):
        self._iterator = iter(self.items)
        self.start_time = time()
        self._processed_count = 0
        if not self.silence: self._update_progress()  # Display the initial (0%) bar
        return self

    def __next__(self):
        # The for-loop protocol will handle the StopIteration from next()
        item = next(self._iterator)

        # The core of the wrapper: call the provided callable for one item.
        result = self.callable(item)
        self._processed_count += 1
        if not self.silence:
            self._update_progress()
        return result

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into a HH:MM:SS string."""
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return "00:00:00"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def _update_progress(self):
        """Calculates and prints the progress bar to stderr."""
        if self.total == 0: return
        percent_complete = self._processed_count / self.total
        filled_length = int(self.bar_length * percent_complete)
        bar = self.bar_character * filled_length + '-' * (self.bar_length - filled_length)
        elapsed_time = time() - self.start_time
        # Calculate Estimated Time of Arrival (ETA)
        if self._processed_count > 0:
            avg_time_per_item = elapsed_time / self._processed_count
            remaining_items = self.total - self._processed_count
            eta = avg_time_per_item * remaining_items
        else: eta = float('inf')
        # Format time strings
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta) if eta != float('inf') else '??:??:??'
        # Use carriage return '\r' to stay on the same line
        stderr.write(
            (f'\r{self.desc}: {int(percent_complete * 100):>3}%|{bar}| '
             f'{self._processed_count}/{self.total} '
             f'[{elapsed_str}<{eta_str}]')
        )
        # When the loop is finished, print a newline to move to the next line
        if self._processed_count == self.total: stderr.write('\n')
        stderr.flush()


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


def download(url: Union[str, Request], dest: Union[str, Path] = None, data = None, encode_data: bool = False) -> Union[Path, bytes, None]:
    """
    Downloads a file from a URL to a destination or returns the data as bytes.

    :param url: URL to download from.
    :param dest: Destination path to save the file to. If None, returns the data as bytes.
    :return: Path to the downloaded file or the data as bytes.
    :raises ValueError: If no data is written to the destination file.
    """
    if data is not None and encode_data: data = urlencode(data).encode('utf-8')
    if not isinstance(url, Request): url = Request(url, headers={'User-Agent': 'Mozilla/5.0'}, data=data)
    with urlopen(url) as response:
        if dest:
            dest = Path(dest)  # stream copy to disk
            with open(dest, 'wb') as f_out: copyfileobj(response, f_out)
            return dest
        else: return response.read()


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
