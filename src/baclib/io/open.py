from io import IOBase
from typing import Union, BinaryIO, Optional
from pathlib import Path
from sys import stdout, stdin
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

        if isinstance(self.file, IOBase): raw_stream = self.file
        elif str(self.file) in {'-', 'stdin'} and 'r' in self.mode: raw_stream = stdin.buffer
        elif str(self.file) in {'-', 'stdout'} and 'w' in self.mode: raw_stream = stdout.buffer
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
                        self._close_on_exit = True # Wrapper needs closing
                        return self._get_opener(pkg)(raw_stream, mode='rb')
                
                if should_close: self._close_on_exit = True
                return raw_stream
        except (AttributeError, ValueError, OSError): pass

        # Non-Seekable (stdin, pipes) -> Use PeekableHandle
        peekable = PeekableHandle(raw_stream)
        start = peekable.peek(self._MIN_N_BYTES)
        
        for magic, pkg in self._MAGIC.items():
            if start.startswith(magic):
                self._close_on_exit = True
                return self._get_opener(pkg)(peekable, mode='rb')
        
        if should_close: self._close_on_exit = True
        return peekable
