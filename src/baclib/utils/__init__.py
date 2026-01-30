"""
Module containing various utility functions and classes.
"""
from json import loads as json_loads
from argparse import Namespace
from dataclasses import dataclass, fields
from io import BytesIO, IOBase
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Union, Iterable, Callable, Any,  BinaryIO, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from shutil import copyfileobj
from time import time
from sys import stderr, stdout, stdin
from importlib import import_module


# Classes --------------------------------------------------------------------------------------------------------------
class PeekableHandle:
    """
    A wrapper around a BinaryIO stream that allows peeking at the beginning of the
    content without consuming it. Used by SeqFile to sniff formats.
    """
    __slots__ = ('_stream', '_peek_buffer', '_buffer_pos', '_buffer_len')

    def __init__(self, stream: BinaryIO, max_peek: int = 4096):
        """
        Initializes the PeekableHandle.

        Args:
            stream: The underlying text stream.
            max_peek: Maximum number of characters to buffer for peeking.
        """
        self._stream = stream
        # Read characters (not bytes) from the BinaryIO stream
        self._peek_buffer = stream.read(max_peek)
        self._buffer_pos = 0
        self._buffer_len = len(self._peek_buffer)

    def peek(self, size: int = -1) -> bytes:
        """
        Returns content from the buffer without advancing the stream position.

        Args:
            size: Number of characters to peek. If -1, returns the entire buffer.

        Returns:
            The peeked string.
        """
        if size == -1 or size > self._buffer_len: return self._peek_buffer
        return self._peek_buffer[:size]

    def read(self, size: int = -1) -> bytes:
        """
        Reads from the stream, consuming the buffer first if available.

        Args:
            size: Number of characters to read. If -1, reads until EOF.

        Returns:
            The read string.
        """
        # 1. Buffer exhausted
        if self._buffer_pos >= self._buffer_len:
            return self._stream.read(size)

        # 2. Read all (rest of buffer + stream)
        if size == -1:
            chunk = self._peek_buffer[self._buffer_pos:]
            self._buffer_pos = self._buffer_len
            return chunk + self._stream.read()

        # 3. Read partial
        available = self._buffer_len - self._buffer_pos
        if size <= available:
            chunk = self._peek_buffer[self._buffer_pos: self._buffer_pos + size]
            self._buffer_pos += size
            return chunk
        else:
            chunk = self._peek_buffer[self._buffer_pos:]
            self._buffer_pos = self._buffer_len
            return chunk + self._stream.read(size - available)

    def __iter__(self):
        """
        Iterates over lines in the stream, handling the buffer seamlessly.

        Yields:
            Lines from the stream.
        """
        # 1. Yield lines from the buffer
        if self._buffer_pos < self._buffer_len:
            fragment = self._peek_buffer[self._buffer_pos:]
            self._buffer_pos = self._buffer_len

            # We use splitlines(keepends=True) to handle line boundaries safely
            lines = fragment.splitlines(keepends=True)

            # If the last segment doesn't end with a newline, it's incomplete.
            # We must stitch it with the next line from the stream.
            for i, line in enumerate(lines):
                if i == len(lines) - 1 and not line.endswith(b'\n'):
                    # Stitch with next line from stream
                    yield line + self._stream.readline()
                else:
                    yield line

        # 2. Yield rest from stream
        yield from self._stream

    def close(self):
        """Closes the underlying stream if possible."""
        if hasattr(self._stream, 'close'): self._stream.close()


class Xopen:
    """
    Handles the Physical Layer: Compression and File System.

    Examples:
        >>> with Xopen("file.gz", "rb") as f:
        ...     content = f.read()
    """
    _MAGIC = {
        b'\x1f\x8b': 'gzip',
        b'\x42\x5a': 'bz2',
        b'\xfd7zXZ\x00': 'lzma',
        b'\x28\xb5\x2f\xfd': 'zstandard'
    }
    _EXT_TO_PKG = {'gz': 'gzip', 'bz2': 'bz2', 'xz': 'lzma', 'zst': 'zstandard'}
    _MIN_N_BYTES = max(len(i) for i in _MAGIC.keys())
    _OPEN_FUNCS = {}

    def __init__(self, file: Union[str, Path, BinaryIO], mode: str = 'rb'):
        """
        Initializes the Xopen context manager.

        Args:
            file: File path (str or Path) or an existing file object.
            mode: File opening mode (e.g., 'rb', 'wb').
        """
        self.file = file
        self.mode = mode
        self._handle: Optional[BinaryIO] = None
        self._close_on_exit = False

    def __enter__(self) -> BinaryIO:
        """
        Opens the file and returns the file handle.

        Returns:
            The opened file handle.
        """
        self._handle = self._open()
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the file handle if it was opened by this instance.
        """
        if self._close_on_exit and self._handle: self._handle.close()

    def _get_opener(self, pkg_name: str):
        """
        Retrieves the open function for a compression package, importing it if necessary.

        Args:
            pkg_name: Name of the compression package (e.g., 'gzip').

        Returns:
            The open function from the package.

        Raises:
            SeqFileError: If the module cannot be imported.
        """
        if pkg_name not in self._OPEN_FUNCS:
            try:
                mod = import_module(pkg_name)
                self._OPEN_FUNCS[pkg_name] = mod.open
            except ImportError:
                raise ModuleNotFoundError(f"Compression module '{pkg_name}' not installed.")
        return self._OPEN_FUNCS[pkg_name]

    def _open(self) -> BinaryIO:
        """
        Internal method to open the file based on type and compression.

        Returns:
            The opened text file handle.
        """
        # 1. Resolve Raw Stream
        raw_stream = None
        should_close = False

        if isinstance(self.file, IOBase):
            raw_stream = self.file
        elif str(self.file) in {'-', 'stdin'} and 'r' in self.mode:
            raw_stream = stdin.buffer
        elif str(self.file) in {'-', 'stdout'} and 'w' in self.mode:
            raw_stream = stdout.buffer
        else:
            # Path
            path = Path(self.file).expanduser()

            # Write mode: Extension based (Fast path)
            if 'w' in self.mode or 'a' in self.mode:
                self._close_on_exit = True
                ext = path.suffix.lower().lstrip('.')
                if pkg := self._EXT_TO_PKG.get(ext): return self._get_opener(pkg)(path, mode=self.mode)
                return open(path, mode=self.mode)

            # Read mode: Open raw
            raw_stream = open(path, mode='rb')
            should_close = True

        # 2. Handle Write Mode (Stream/Stdout)
        if 'w' in self.mode or 'a' in self.mode: return raw_stream

        # 3. Handle Read Mode: Sniff Compression
        # Try Seekable (Fast Path for Files)
        try:
            if raw_stream.seekable():
                start = raw_stream.read(self._MIN_N_BYTES)
                raw_stream.seek(0)
                for magic, pkg in self._MAGIC.items():
                    if start.startswith(magic):
                        self._close_on_exit = True  # Wrapper needs closing
                        return self._get_opener(pkg)(raw_stream, mode='rb')

                if should_close: self._close_on_exit = True
                return raw_stream
        except (AttributeError, ValueError, OSError):
            pass

        # Non-Seekable (stdin, pipes) -> Use PeekableHandle
        peekable = PeekableHandle(raw_stream)
        start = peekable.peek(self._MIN_N_BYTES)

        for magic, pkg in self._MAGIC.items():
            if start.startswith(magic):
                self._close_on_exit = True
                return self._get_opener(pkg)(peekable, mode='rb')

        if should_close: self._close_on_exit = True
        return peekable


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



# def otsu(similarity_matrix) -> float:
#     """
#     Calculates Otsu's threshold for a similarity matrix.
#
#     Args:
#         similarity_matrix (np.ndarray): Array of similarity scores.
#
#     Returns:
#         float: The calculated threshold.
#     """
#     if len(similarity_matrix) == 0: return 0.5
#     hist, bin_edges = np.histogram(similarity_matrix, bins=256, range=(0.0, 1.0))
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     weight = hist.cumsum()
#     mean = (hist * bin_centers).cumsum()
#     total_mean = mean[-1]
#     total_weight = weight[-1]
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mean_bg = mean / weight
#         mean_fg = (total_mean - mean) / (total_weight - weight)
#     mean_bg[np.isnan(mean_bg)] = 0.0
#     mean_fg[np.isnan(mean_fg)] = 0.0
#     w0 = weight
#     w1 = total_weight - weight
#     between_class_variance = w0 * w1 * (mean_bg - mean_fg) ** 2
#     idx = np.argmax(between_class_variance)
#     return bin_centers[idx]
