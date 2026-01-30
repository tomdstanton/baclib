"""
Module for managing external programs such as Minimap2.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Generator, Union, Literal, Optional, Any, IO
from subprocess import Popen, PIPE, DEVNULL
from dataclasses import dataclass, fields
from threading import Thread
import errno

from baclib.core.interval import Interval
from baclib.core.seq import Alphabet
from baclib.containers.record import Record, Feature
from baclib.io.align import PafReader
from baclib.io.seq import FastaWriter, FastaReader
from baclib.utils import Config, LiteralFile
from baclib.utils.resources import RESOURCES


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class ExternalProgramError(Exception): pass
class Minimap2Error(ExternalProgramError): pass


# Classes --------------------------------------------------------------------------------------------------------------
class _ProcessStream:
    """
    Wraps a subprocess stdout to behave like a file-like object (BinaryIO),
    ensuring the process and writer thread are cleaned up correctly on close.
    """
    def __init__(self, proc: Popen, writer_thread: Thread, write_error: list, program: str):
        self._proc = proc
        self._thread = writer_thread
        self._write_error = write_error
        self._program = program
        self._stdout = proc.stdout
        self._closed = False

    def read(self, n: int = -1) -> bytes:
        return self._stdout.read(n)

    def readline(self, limit: int = -1) -> bytes:
        return self._stdout.readline(limit)

    def __iter__(self):
        return self._stdout

    def close(self):
        if self._closed: return
        self._closed = True

        # Prevent deadlocks: If the process is still running, it might be blocked writing to us.
        terminated = False
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                terminated = True
            except OSError: pass

        # Wait for process to finish
        self._proc.wait()
        self._thread.join()

        # Check for errors
        if not terminated and self._proc.returncode != 0:
            stderr = self._proc.stderr.read()
            raise ExternalProgramError(f"{self._program} failed (code {self._proc.returncode}): {stderr.decode('utf-8', errors='replace')}")

        if self._write_error: raise self._write_error[0]

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()


class ExternalProgram:
    """
    Base class to handle an external program to be executed in subprocesses.

    This class provides a robust way to run external command-line tools,
    handling process creation, streaming I/O, and error reporting. It is
    optimized to avoid using `shell=True` for security and performance.
    """
    def __init__(self, program: str):
        if not (binary := RESOURCES.find_binary(program)):
            raise ExternalProgramError(f'Could not find {program}')
        self._program = program
        self._binary = binary

    def __repr__(self): return f'{self._program}({self._binary})'

    @property
    def version(self) -> bytes:
        try:
            stdout, _ = self.run(['--version'])
            return stdout.strip().split(maxsplit=1)[-1]
        except Exception: return b"unknown"

    def run(self, args: list[str], input_: bytes = None) -> tuple[bytes, bytes]:
        """Blocking execution (for short tasks like indexing/version)."""
        cmd = [self._binary] + args
        with Popen(cmd, stdin=PIPE if input_ else DEVNULL, stdout=PIPE, stderr=PIPE) as proc:
            return proc.communicate(input=input_)

    def stream(self, args: list[str]) -> Generator[bytes, None, None]:
        """
        Streaming execution (for long tasks like alignment).
        Yields lines from stdout as they become available.
        """
        cmd = [self._binary] + args
        with Popen(cmd, stdout=PIPE, stderr=PIPE) as proc:
            yield from proc.stdout

            # Wait for process to finish *after* reading all stdout
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read()
                raise ExternalProgramError(f"{self._program} failed (code {proc.returncode}): {stderr.decode('utf-8', errors='replace')}")

    def _stream_input_output(self, args: list[str], input_items: Iterable[Any], writer_cls=FastaWriter) -> IO[bytes]:
        """
        Advanced streaming: Writes items to subprocess stdin using a Writer class in a background thread,
        while returning a file-like object of the subprocess stdout.
        """
        cmd = [self._binary] + args
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        # Container to propagate exceptions from the thread
        write_error = []

        def writer_thread():
            try:
                # writer_cls (e.g. FastaWriter) wraps the pipe
                with writer_cls(proc.stdin) as writer:
                    for item in input_items:
                        writer.write_one(item)
            except BrokenPipeError:
                # Process died early or closed stdin; stop writing silently.
                # The main thread will handle the process exit code/stderr.
                pass
            except OSError as e:
                if e.errno == errno.EPIPE:
                    pass
                else:
                    write_error.append(e)
            except Exception as e:
                write_error.append(e)
            finally:
                # Ideally FastaWriter closes it, but we force close to be safe
                # so the subprocess knows input is done.
                try: proc.stdin.close()
                except (OSError, ValueError): pass

        # Start the writer thread
        t = Thread(target=writer_thread)
        t.start()

        return _ProcessStream(proc, t, write_error, self._program)

    @staticmethod
    def _build_params(config: Config) -> list[str]:
        params = []
        for field in fields(config):
            key = field.name
            val = getattr(config, key)
            if val is None or val is False: continue
            flag = f"-{key}" if len(key) == 1 else f"--{key.replace('_', '-')}"
            params.append(flag)
            if val is not True:
                params.append(str(val))
        return params


@dataclass
class Minimap2IndexConfig(Config):
    t: int = RESOURCES.available_cpus
    H: bool = False
    k: int = None
    w: int = None
    I: str = None


@dataclass
class Minimap2AlignConfig(Config):
    x: str = None
    t: int = RESOURCES.available_cpus
    f: float = None
    g: int = None
    G: int = None
    F: int = None
    r: int = None
    n: int = None
    m: int = None
    X: bool = False
    p: float = None
    N: int = None
    A: int = None
    B: int = None
    O: int = None
    E: int = None
    z: int = None
    s: int = None
    u: str = None
    J: int = None
    a: bool = False
    c: bool = True
    cs: bool = False
    ds: bool = False
    MD: bool = False
    eqx: bool = False


class Minimap2(ExternalProgram):
    """
    A wrapper for the external program Minimap2.

    This class handles building indices and running alignments by streaming
    `baclib.Record` objects to and from the Minimap2 subprocess, avoiding
    the need to write intermediate files to disk.

    Requires `minimap2` to be in the system's PATH.

    Examples:
        >>> mm2 = Minimap2()
        >>> mm2.build_index(target_record)
        >>> for hit in mm2.align(query_record):
        ...     print(hit)
    """
    _ALPHABET = Alphabet.dna()
    _DEFAULT_ALIGN_CONFIG = Minimap2AlignConfig()
    _DEFAULT_INDEX_CONFIG = Minimap2IndexConfig()

    def __init__(self, *targets: Record, align_config: Minimap2AlignConfig = None, index_config: Minimap2IndexConfig = None):
        super().__init__('minimap2')
        self._target_index: Optional[Path] = None
        self._temp_files: list[Path] = []
        self._align_config = align_config or self._DEFAULT_ALIGN_CONFIG
        self._index_config = index_config or self._DEFAULT_INDEX_CONFIG
        if targets:
            self.build_index(*targets)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.cleanup()
    def __del__(self): self.cleanup()
    def __call__(self, *args, **kwargs): return self.align(*args, **kwargs)

    def cleanup(self):
        """Cleans up index and any materialized sequence files."""
        if self._target_index:
            try:
                if hasattr(self._target_index, 'unlink'): self._target_index.unlink(missing_ok=True)
                elif isinstance(self._target_index, str): Path(self._target_index).unlink(missing_ok=True)
            except OSError: pass
            self._target_index = None

        for p in self._temp_files:
            try: p.unlink(missing_ok=True)
            except OSError: pass
        self._temp_files.clear()

    def build_index(self, *targets: Union[str, Path, Record], config: Minimap2IndexConfig = None) -> Path:
        """
        Builds a Minimap2 index.

        Args:
            *targets: Target sequences (Records or file paths).
            config: Indexing configuration.

        Returns:
            Path to the index file.
        """
        if not config: config = self._index_config

        # Create a named temp file for the output index
        tf = NamedTemporaryFile(suffix='.mmi', delete=False)
        tf.close()
        index_path = Path(tf.name)

        params = self._build_params(config)

        # Check if targets are Paths/Strings (already on disk) or Records (need streaming)
        all_paths = True
        for t in targets:
            if not isinstance(t, (str, Path)):
                all_paths = False
                break
            if not Path(t).exists():
                all_paths = False
                break

        if all_paths:
            args = params + ['-d', str(index_path)] + [str(t) for t in targets]
            _, stderr = self.run(args)
        else:
            # Stream records to stdin ('-')
            args = params + ['-d', str(index_path), '-']
            # Consume generator locally
            with self._stream_input_output(args, targets, lambda f: FastaWriter(f, width=0)) as stream:
                for _ in stream: pass

        if not LiteralFile.from_path(index_path, return_bool=True):
            raise Minimap2Error(f'Failed to build index at {index_path}')

        self._target_index = index_path
        self._temp_files.append(index_path)
        return index_path

    def align(self, *queries: Record, targets: Iterable[Record] = (), config: Minimap2AlignConfig = None):
        """
        Aligns query sequences to the target index.

        Args:
            *queries: Query records.
            targets: Optional targets (if index not already built).
            config: Alignment configuration.

        Yields:
            Alignment objects (parsed from PAF).
        """
        if not config: config = self._align_config

        target_arg = None
        if targets:
            target_arg = str(self.build_index(*targets))
        elif self._target_index:
            target_arg = str(self._target_index)
        else:
            raise Minimap2Error('Targets must be supplied if no target index has been built')

        params = self._build_params(config)
        args = params + [target_arg, '-']

        with self._stream_input_output(args, queries, lambda f: FastaWriter(f, width=0)) as stream:
            yield from PafReader(stream)


@dataclass
class FragGeneScanRsConfig(Config):
    thread_num: int = RESOURCES.available_cpus
    formatted: bool = False
    unordered: bool = True
    complete: bool = True
    training_file: Literal['complete', 'sanger_5', 'sanger_10', '454_5', '454_10', '454_30', 'illumina_1',
    'illumina_5', 'illumina_10'] = 'complete'


class FragGeneScanRsError(ExternalProgramError): pass


class FragGeneScanRs(ExternalProgram):
    """
    A wrapper for the external program FragGeneScanRs for gene prediction.

    Streams sequence records to the FragGeneScanRs subprocess and parses
    the predicted protein sequences, adding them as features to the original
    nucleotide records.

    Requires `FragGeneScanRs` to be in the system's PATH.

    Examples:
        >>> fgs = FragGeneScanRs()
        >>> for record in fgs.predict(dna_record):
        ...     print(len(record.features))
    """
    _ALPHABET = Alphabet.amino()
    _DEFAULT_CONFIG = FragGeneScanRsConfig()

    def __init__(self, config: FragGeneScanRsConfig = None):
        super().__init__('FragGeneScanRs')
        self._config = config or self._DEFAULT_CONFIG

    def predict(self, *seqs: Record, config: FragGeneScanRsConfig = None) -> Generator[Record, None, None]:
        """
        Predicts genes in the given sequences.

        Args:
            *seqs: Input DNA records.
            config: Configuration options.

        Yields:
            Protein records (representing the predicted CDS translations).
            Side effect: Adds CDS features to the input `seqs` objects.
        """
        config = config or self._config
        args = self._build_params(config) + ['--core-file-name', 'stdin']

        seq_map = {i.id: i for i in seqs}

        with self._stream_input_output(args, seqs, lambda f: FastaWriter(f, width=0)) as stream:
            for record in FastaReader(stream, alphabet=self._ALPHABET):
                try:
                    # FragGeneScanRs header: >parent_start_end_strand
                    parent_id, start, end, strand = record.id.rsplit(b'_', 3)
                    if parent_id in seq_map:
                        seq_map[parent_id].features.append(
                            Feature(
                                Interval(int(start) - 1, int(end), strand),
                                kind=b'CDS',
                                qualifiers=[
                                    (b'transl_table', 11),
                                    (b'inference', b'ab initio prediction:FragGeneScanRs:' + self.version),
                                ]
                            )
                        )
                except ValueError:
                    # Robustness: Don't crash if FGS outputs a weird header, just skip associating the feature
                    pass

                yield record
