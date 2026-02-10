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

from baclib.core.interval import Interval, Strand
from baclib.core.seq import Alphabet
from baclib.containers.record import Record, RecordBatch, Feature, FeatureKey
from baclib.io.tabular import PafReader
from baclib.io.seq import FastaWriter, FastaReader
from baclib.utils import Config, LiteralFile, Batch
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
    __slots__ = ('_program', '_binary')

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
                        writer.write(item)
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

    @staticmethod
    def _normalize_input(items: Union[Any, Iterable[Any]]) -> Iterable[Any]:
        """
        Normalizes input to an iterable.
        Handles single Record, Batch, str, or Path by wrapping them in a list.
        """
        if items is None: return []
        if isinstance(items, (str, Path, Record)): return [items]
        return items

    @staticmethod
    def _flatten_input(items: Iterable[Union[Record, Batch]]) -> Generator[Record, None, None]:
        """Flattens an iterable of Records and Batches into a generator of Records."""
        for item in items:
            if isinstance(item, Batch): yield from item
            else: yield item


@dataclass(slots=True, frozen=True, kw_only=True)
class Minimap2IndexConfig(Config):
    t: int = RESOURCES.available_cpus
    H: bool = False
    k: int = None
    w: int = None
    I: str = None


@dataclass(slots=True, frozen=True, kw_only=True)
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
    _ALPHABET = Alphabet.DNA
    _DEFAULT_ALIGN_CONFIG = Minimap2AlignConfig()
    _DEFAULT_INDEX_CONFIG = Minimap2IndexConfig()
    __slots__ = ('_target_index', '_temp_files', '_align_config', '_index_config')

    def __init__(self, targets: Union[Record, Batch, Iterable] = None, align_config: Minimap2AlignConfig = None, index_config: Minimap2IndexConfig = None):
        super().__init__('minimap2')
        self._target_index: Optional[Path] = None
        self._temp_files: list[Path] = []
        self._align_config = align_config or self._DEFAULT_ALIGN_CONFIG
        self._index_config = index_config or self._DEFAULT_INDEX_CONFIG
        if targets:
            self.build_index(targets)

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

    def build_index(self, targets: Union[str, Path, Record, Batch, Iterable], config: Minimap2IndexConfig = None) -> Path:
        """
        Builds a Minimap2 index.

        Args:
            targets: Target sequences (Records, Batches, file paths, or iterable thereof).
            config: Indexing configuration.

        Returns:
            Path to the index file.
        """
        if not config: config = self._index_config

        targets = self._normalize_input(targets)

        # Create a named temp file for the output index
        tf = NamedTemporaryFile(suffix='.mmi', delete=False)
        tf.close()
        index_path = Path(tf.name)

        params = self._build_params(config)

        # Determine if we can use file paths directly or need to stream.
        # We only use file paths if the input is a re-iterable collection (list/tuple)
        # and all elements are valid paths. If it's a generator or Batch, we stream.
        use_files = False
        if isinstance(targets, (list, tuple)):
            # Check if non-empty and first element looks like a path
            if targets and isinstance(targets[0], (str, Path)):
                if all(isinstance(t, (str, Path)) and Path(t).exists() for t in targets):
                    use_files = True

        if use_files:
            args = params + ['-d', str(index_path)] + [str(t) for t in targets]
            _, stderr = self.run(args)
        else:
            # Stream records to stdin ('-')
            # Note: If targets is a generator, _flatten_input consumes it.
            # This is fine as we don't reuse it here.
            args = params + ['-d', str(index_path), '-']
            # Consume generator locally
            with self._stream_input_output(args, self._flatten_input(targets), lambda f: FastaWriter(f, width=0, threaded=False)) as stream:
                for _ in stream: pass

        index_path = LiteralFile(index_path)
        if not index_path: raise Minimap2Error(f'Failed to build index at {index_path}')

        self._target_index = index_path
        self._temp_files.append(index_path)
        return index_path

    def align(self, queries: Union[Record, Batch, Iterable], targets: Union[Record, Batch, Iterable] = None, config: Minimap2AlignConfig = None):
        """
        Aligns query sequences to the target index.

        Args:
            queries: Query records (Record, Batch, or Iterable).
            targets: Optional targets (if index not already built).
            config: Alignment configuration.

        Yields:
            Alignment objects (parsed from PAF).
        """
        if not config: config = self._align_config

        queries = self._normalize_input(queries)

        target_arg = None
        if targets:
            target_arg = str(self.build_index(targets))
        elif self._target_index:
            target_arg = str(self._target_index)
        else:
            raise Minimap2Error('Targets must be supplied if no target index has been built')

        params = self._build_params(config)
        args = params + [target_arg, '-']

        with self._stream_input_output(args, self._flatten_input(queries), lambda f: FastaWriter(f, width=0, threaded=False)) as stream:
            yield from PafReader(stream)


@dataclass(slots=True, frozen=True, kw_only=True)
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
    the predicted protein sequences.

    Requires `FragGeneScanRs` to be in the system's PATH.


    """
    _ALPHABET = Alphabet.AMINO
    _DEFAULT_CONFIG = FragGeneScanRsConfig()
    __slots__ = ('_config',)

    def __init__(self, config: FragGeneScanRsConfig = None):
        super().__init__('FragGeneScanRs')
        self._config = config or self._DEFAULT_CONFIG

    def predict(self, seqs: Union[Record, Batch, Iterable], config: FragGeneScanRsConfig = None) -> RecordBatch:
        """
        Predicts genes in the given sequences.

        Args:
            seqs: Input DNA records (Record, Batch, or Iterable).
            config: Configuration options.

        Returns:
            A new RecordBatch containing the input records with added CDS features.
            The CDS features include a 'translation' qualifier with the protein sequence.
        """
        config = config or self._config
        
        # Optimization: If input is a RecordBatch, we can avoid re-batching sequences later
        input_seq_batch = seqs.seqs if isinstance(seqs, RecordBatch) else None

        # Materialize records to attach features
        records = list(self._flatten_input(self._normalize_input(seqs)))
        if not records: return RecordBatch([])
        
        rec_map = {r.id: r for r in records}

        args = self._build_params(config) + ['--seq-file-name', 'stdin']
        
        # Pre-compute constant values to avoid overhead in the loop
        inference_val = b'ab initio prediction:FragGeneScanRs: ' + self.version
        
        # Stream records to FGS and parse output
        with self._stream_input_output(args, records, lambda f: FastaWriter(f, width=0, threaded=False)) as stream:
            # FGS output is unwrapped (1 line sequence)
            reader = FastaReader(stream, alphabet=self._ALPHABET, n_seq_lines=1)
            
            for batch in reader.batches():
                ids = batch.ids
                prot_seqs = batch.seqs
                
                # Optimization: Cache local lookups
                last_parent_id = None
                last_record = None
                
                for i in range(len(batch)):
                    # Parse ID: parent_start_end_strand
                    full_id = ids[i]
                    try:
                        parent_id, start, end, strand_char = full_id.rsplit(b'_', 3)
                    except ValueError:
                        continue
                    
                    # Avoid dict lookup if parent hasn't changed (FGS groups output by input seq)
                    if parent_id == last_parent_id:
                        record = last_record
                    else:
                        record = rec_map.get(parent_id)
                        last_parent_id = parent_id
                        last_record = record

                    if record:
                        # FGS is 1-based inclusive, Interval is 0-based half-open
                        # Use Strand enum directly to skip from_symbol check
                        iv = Interval(int(start) - 1, int(end), Strand.from_bytes(strand_char))
                        
                        feat = Feature(
                            iv,
                            key=FeatureKey.CDS,
                            qualifiers=[
                                (b'transl_table', 11),
                                (b'inference', inference_val),
                                (b'translation', prot_seqs[i])
                            ]
                        )
                        record.features.append(feat)

        # Reconstruct RecordBatch
        if input_seq_batch and len(records) == len(input_seq_batch):
            return RecordBatch.from_aligned_batch(input_seq_batch, records)
        
        return RecordBatch(records)
