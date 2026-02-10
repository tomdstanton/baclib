"""
Module containing various utility functions and classes.
"""
from abc import ABC, abstractmethod
from json import loads as json_loads
from dataclasses import dataclass, fields
from io import IOBase
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Union, Iterable, BinaryIO, Optional, IO, Any
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from shutil import copyfileobj, get_terminal_size
from time import time
from sys import stderr, stdout, stdin
import threading
import queue
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .resources import RESOURCES


# Classes --------------------------------------------------------------------------------------------------------------
class Batch(ABC):
    """
    Abstract base class for all batch containers.
    Enforces the Sequence protocol (len, getitem, iter).
    """
    __slots__ = ()
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def empty(self) -> 'Batch': ...
    @abstractmethod
    def __getitem__(self, item): ...
    def __iter__(self):
        for i in range(len(self)): yield self[i]


class RaggedBatch(Batch):
    """
    Base class for batches that store variable-length items in a flattened format (CSR-like).
    Manages the offsets array and length calculation.
    """
    __slots__ = ('_offsets', '_length')
    def __init__(self, offsets: np.ndarray): 
        self._offsets = offsets
        self._length = len(offsets) - 1
    def __len__(self) -> int: return self._length

    def empty(self) -> 'RaggedBatch':
        return self.__class__(np.array([0], dtype=self._offsets.dtype))

    def _get_slice_info(self, item: slice) -> tuple[np.ndarray, int, int]:
        start, stop, step = item.indices(len(self))
        if step != 1: raise NotImplementedError("Batch slicing with step != 1 not supported")
        val_start = self._offsets[start]
        val_end = self._offsets[stop]
        new_offsets = self._offsets[start:stop+1] - val_start
        return new_offsets, val_start, val_end


class ThreadedChunkReader:
    """
    Reads data from a file handle in a background thread and yields chunks.
    Useful for hiding IO latency (e.g. decompression) during parsing.
    """
    def __init__(self, handle: BinaryIO, chunk_size: int = 65536, queue_size: int = 4):
        self._handle = handle
        self._chunk_size = chunk_size
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while True:
                chunk = self._handle.read(self._chunk_size)
                self._queue.put(chunk)
                if not chunk: break
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if isinstance(item, Exception): raise item
            yield item
            if not item: break


class ThreadedChunkWriter:
    """
    Writes data to a file handle in a background thread.
    Buffers small writes into chunks to reduce queue overhead and lock contention.
    """
    def __init__(self, handle: BinaryIO, chunk_size: int = 65536, queue_size: int = 4):
        self._handle = handle
        self._chunk_size = chunk_size
        self._queue = queue.Queue(maxsize=queue_size)
        self._buffer = []
        self._buffer_len = 0
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while True:
                chunk = self._queue.get()
                if chunk is None: break
                self._handle.write(chunk)
                self._queue.task_done()
        except Exception as e:
            self._error = e

    def write(self, data: bytes):
        if self._error: raise self._error
        self._buffer.append(data)
        self._buffer_len += len(data)
        if self._buffer_len >= self._chunk_size:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer: return
        data = b"".join(self._buffer)
        self._buffer = []
        self._buffer_len = 0
        self._queue.put(data)

    def close(self):
        self._flush_buffer()
        self._queue.put(None)
        self._thread.join()
        if self._error: raise self._error


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

        # Non-Seekable (stdin, pipes)
        # Optimization: Use native peek if available (e.g. BufferedReader) to avoid Python-level buffering overhead
        stream_to_use = raw_stream
        if hasattr(raw_stream, 'peek'):
            start = raw_stream.peek(self._MIN_N_BYTES)[:self._MIN_N_BYTES]
        else:
            stream_to_use = PeekableHandle(raw_stream)
            start = stream_to_use.peek(self._MIN_N_BYTES)

        for magic, pkg in self._MAGIC.items():
            if start.startswith(magic):
                self._close_on_exit = True
                return self._get_opener(pkg)(stream_to_use, mode='rb')

        if should_close: self._close_on_exit = True
        return stream_to_use


@dataclass(slots=True, frozen=True, kw_only=True)
class Config:
    """
    Config parent class that can conveniently set attributes from CLI args
    """
    @classmethod
    def from_obj(cls, obj: Any) -> 'Config':
        return cls(**{f.name: val for f in fields(cls) if (val := getattr(obj, f.name, None)) is not None})


class LiteralFile(type(Path())):
    """
    A Path wrapper that evaluates to False if the file is missing or empty.
    Inherits from the concrete Path type (PosixPath/WindowsPath) to ensure correct instantiation.
    """
    _MIN_SIZE = 1
    
    def __bool__(self):
        try: return self.is_file() and self.stat().st_size >= self._MIN_SIZE
        except OSError: return False


class Downloader:
    """
    High-performance downloader supporting parallel chunk retrieval.
    """
    _MAX_WORKERS = (RESOURCES.available_cpus or 1) * 4
    _CHUNK_SIZE = 10 * 1024 * 1024

    def __enter__(self):
        self._pool = ThreadPoolExecutor(max_workers=self._MAX_WORKERS)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_pool'):
            self._pool.shutdown(wait=True)
            del self._pool

    def fetch(self, url: Union[str, Request], dest: Union[str, Path] = None, data=None, 
              encode_data: bool = False) -> Union[Path, bytes]:
        if data is not None and encode_data: data = urlencode(data).encode('utf-8')
        if not isinstance(url, Request): 
            url = Request(url, headers={'User-Agent': 'Mozilla/5.0'}, data=data)

        # Fallback for POST or small files
        if url.data is not None:
            return self._single_thread_download(url, dest)

        try:
            size, accept_ranges, real_url = self._get_info(url)
        except Exception:
            return self._single_thread_download(url, dest)

        if accept_ranges and size and size > self._CHUNK_SIZE:
            return self._parallel_download(real_url, dest, size, url.headers)
        
        return self._single_thread_download(url, dest)
    
    @staticmethod
    def _get_info(req: Request):
        head_req = Request(req.full_url, headers=req.headers, method='HEAD')
        with urlopen(head_req) as response:
            size = int(response.headers.get('Content-Length', 0))
            accept_ranges = response.headers.get('Accept-Ranges', 'none') == 'bytes'
            return size, accept_ranges, response.geturl()
    
    @staticmethod
    def _single_thread_download(url, dest):
        with urlopen(url) as response:
            if dest:
                dest = Path(dest)
                with open(dest, 'wb') as f: copyfileobj(response, f)
                return dest
            return response.read()
    
    
    def _parallel_download(self, url_str, dest, size, headers):
        chunks = []
        for i in range(0, size, self._CHUNK_SIZE):
            start = i
            end = min(i + self._CHUNK_SIZE - 1, size - 1)
            chunks.append((start, end))

        if dest:
            dest = Path(dest)
            # Create sparse file
            with open(dest, 'wb') as f: f.truncate(size)
            
            def _worker(start, end):
                req = Request(url_str, headers=headers)
                req.add_header('Range', f'bytes={start}-{end}')
                with urlopen(req) as response:
                    data = response.read()
                with open(dest, 'r+b') as f:
                    f.seek(start)
                    f.write(data)
        else:
            buffer = bytearray(size)
            def _worker(start, end):
                req = Request(url_str, headers=headers)
                req.add_header('Range', f'bytes={start}-{end}')
                with urlopen(req) as response:
                    data = response.read()
                buffer[start:end+1] = data

        # Use existing pool if available (Context Manager), else create one
        pool = getattr(self, '_pool', None)
        if pool:
            futures = [pool.submit(_worker, s, e) for s, e in chunks]
            for f in as_completed(futures): f.result()
        else:
            with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as pool:
                futures = [pool.submit(_worker, s, e) for s, e in chunks]
                for f in as_completed(futures): f.result()

        return dest if dest else bytes(buffer)


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
        if self._meta_cache is None: self._meta_cache = json_loads(Downloader().fetch(self._api_url))
        return self._meta_cache

    def clone(self) -> Path:
        """
        Downloads and extracts the repo to a temporary directory.
        Returns the Path to the actual source code (inside the extracted folder).
        """
        # If already cloned, just return the path
        if self.local_path and self.local_path.exists(): return self.local_path
        zip_url = f'{self._api_url}/zipball/{self.branch}'
        
        # Create the temp directory explicitly
        self._temp_dir_obj = TemporaryDirectory()
        base_temp_path = Path(self._temp_dir_obj.name)
        zip_path = base_temp_path / "repo.zip"
        with Downloader() as dl:
            dl.fetch(zip_url, zip_path)
        
        with ZipFile(zip_path, 'r') as zfile:
            zfile.extractall(base_temp_path)
            root_folder = zfile.namelist()[0].split('/')[0]
        zip_path.unlink()
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
    A high-performance, zero-overhead progress bar.
    Supports automatic iteration, manual updates, and byte-counting.
    """
    __slots__ = ('_iterable', '_total', '_desc', '_unit', '_scale', '_leave',
                 '_file', '_cols', '_min_interval', '_last_print_t', '_start_t',
                 '_n', '_last_n', '_avg_rate', '_smoothing', '_bar_char', '_lock')

    def __init__(self, iterable: Iterable = None, total: int = None, desc: str = None, 
                 unit: str = 'it', scale: bool = False, leave: bool = True, 
                 file: IO = stderr, min_interval: float = 0.1, bar_char: str = '█',
                 cols: int = None):
        self._iterable = iterable
        self._total = total
        if total is None and iterable is not None:
            try: self._total = len(iterable)
            except (TypeError, AttributeError): pass
        
        self._desc = desc + ": " if desc else ""
        self._unit = unit
        self._scale = scale
        self._leave = leave
        self._file = file
        self._min_interval = min_interval
        self._bar_char = bar_char
        self._cols = cols
        
        self._n = 0
        self._last_n = 0
        self._start_t = time()
        self._last_print_t = self._start_t
        self._avg_rate = 0.0
        self._smoothing = 0.3
        self._lock = threading.Lock()

    def __iter__(self):
        if self._iterable is None: return
        
        # Localize for speed
        n = self._n
        min_int = self._min_interval
        last_t = self._last_print_t
        
        for item in self._iterable:
            yield item
            n += 1
            self._n = n
            curr_t = time()
            if curr_t - last_t >= min_int:
                self._update(curr_t)
                last_t = curr_t
        
        self._update(time(), final=True)

    def __enter__(self):
        self._start_t = time()
        self._last_print_t = self._start_t
        self._update(self._start_t)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._update(time(), final=True)

    def update(self, n: int = 1):
        with self._lock:
            self._n += n
            curr_t = time()
            if curr_t - self._last_print_t >= self._min_interval:
                self._update(curr_t)

    def _format_num(self, n):
        if not self._scale: return str(int(n))
        factor = 1024.0 if self._unit == 'B' else 1000.0
        if n < factor: return f"{n:.0f}"
        for u in ['k', 'M', 'G', 'T', 'P']:
            n /= factor
            if n < factor: return f"{n:.1f}{u}"
        return f"{n:.1f}E"

    def _format_time(self, seconds):
        if not seconds or seconds < 0 or seconds == float('inf'): return "??:??"
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h: return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _update(self, curr_t, final=False):
        dt = curr_t - self._last_print_t
        dn = self._n - self._last_n
        
        # Rate calculation (EMA)
        if dt > 0 and dn >= 0:
            curr_rate = dn / dt
            if self._avg_rate == 0: self._avg_rate = curr_rate
            else: self._avg_rate = (self._avg_rate * (1 - self._smoothing)) + (curr_rate * self._smoothing)
        
        self._last_print_t = curr_t
        self._last_n = self._n

        # Dimensions
        cols = self._cols or get_terminal_size().columns
        
        # Stats
        elapsed = curr_t - self._start_t
        rate_str = self._format_num(self._avg_rate) if self._avg_rate else "?"
        
        if self._total:
            frac = self._n / self._total
            pct = frac * 100
            rem = self._total - self._n
            eta = rem / self._avg_rate if self._avg_rate > 0 else float('inf')
            
            l_bar = f"{self._desc}{pct:3.0f}%|"
            r_bar = f"| {self._format_num(self._n)}/{self._format_num(self._total)} [{self._format_time(elapsed)}<{self._format_time(eta)}, {rate_str}{self._unit}/s]"
            
            bar_len = max(1, cols - len(l_bar) - len(r_bar) - 1)
            fill = int(frac * bar_len)
            bar = self._bar_char * fill + '-' * (bar_len - fill)
            
            line = f"\r{l_bar}{bar}{r_bar}"
        else:
            # Indeterminate
            l_bar = f"{self._desc}"
            r_bar = f" {self._format_num(self._n)} [{self._format_time(elapsed)}, {rate_str}{self._unit}/s]"
            line = f"\r{l_bar}{r_bar}"

        # Pad to clear previous
        if len(line) < cols: line += " " * (cols - len(line))
        
        self._file.write(line)
        self._file.flush()
        
        if final:
            if self._leave: self._file.write('\n')
            else: self._file.write(f"\r{' ' * cols}\r")
            self._file.flush()
