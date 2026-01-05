"""
Module for managing external programs such as Minimap2.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Generator, Union, Literal, Optional, Any
from subprocess import Popen, PIPE, DEVNULL
from dataclasses import dataclass, asdict
from threading import Thread
import errno

from . import RESOURCES
from .seq import Record, Feature, Qualifier, Interval, Alphabet
from .io import PafReader, FastaWriter, FastaReader
from .utils import Config, find_executable_binaries, is_non_empty_file
from .alignment import Alignment


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class ExternalProgramError(Exception): pass
class Minimap2Error(ExternalProgramError): pass


# Classes --------------------------------------------------------------------------------------------------------------
class ExternalProgram:
    """
    Base class to handle an external program to be executed in subprocesses.
    Optimized to avoid shell=True and handle streaming.
    """
    def __init__(self, program: str):
        if not (binary := next(find_executable_binaries(program), None)):
            raise ExternalProgramError(f'Could not find {program}')
        self._program = program
        self._binary = binary
        self._version = None

    def __repr__(self): return f'{self._program}({self._binary})'

    def version(self) -> str:
        if not self._version:
            try:
                stdout, _ = self.run(['--version'])
                self._version = stdout.strip().split(maxsplit=1)[-1]
            except Exception:
                self._version = "unknown"
        return self._version

    def run(self, args: list[str], input_: str = None) -> tuple[str, str]:
        """Blocking execution (for short tasks like indexing/version)."""
        cmd = [self._binary] + args
        with Popen(cmd, stdin=PIPE if input_ else DEVNULL, stdout=PIPE, stderr=PIPE, text=True) as proc:
            return proc.communicate(input=input_)

    def stream(self, args: list[str]) -> Generator[str, None, None]:
        """
        Streaming execution (for long tasks like alignment).
        Yields lines from stdout as they become available.
        """
        cmd = [self._binary] + args
        with Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, bufsize=1) as proc:
            yield from proc.stdout

            # Wait for process to finish *after* reading all stdout
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read()
                raise ExternalProgramError(f"{self._program} failed (code {proc.returncode}): {stderr}")

    def _stream_input_output(self, args: list[str], input_items: Iterable[Any], writer_cls=FastaWriter) -> Generator[str, None, None]:
        """
        Advanced streaming: Writes items to subprocess stdin using a Writer class in a background thread,
        while simultaneously yielding lines from subprocess stdout.
        """
        cmd = [self._binary] + args
        # bufsize=1 for line buffering
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)

        # Container to propagate exceptions from the thread
        write_error = []

        def writer_thread():
            try:
                # writer_cls (e.g. FastaWriter) wraps the pipe
                with writer_cls(proc.stdin) as writer:
                    writer.write(*input_items)
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

        # Yield stdout as it comes (Main Thread)
        yield from proc.stdout

        # Clean up
        proc.wait()
        t.join()

        # If the process failed, prefer its error over the thread's error
        # (because the thread error might just be a symptom of the process crashing)
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise ExternalProgramError(f"{self._program} failed (code {proc.returncode}): {stderr}")

        # If process succeeded but thread had an error (e.g. serialization issue)
        if write_error:
            raise write_error[0]


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
        if not config: config = self._index_config

        # Create a named temp file for the output index
        tf = NamedTemporaryFile(suffix='.mmi', delete=False)
        tf.close()
        index_path = Path(tf.name)

        params = _build_params(config)

        # Check if targets are Paths/Strings (already on disk) or Records (need streaming)
        all_paths = all(isinstance(t, (str, Path)) and Path(t).exists() for t in targets)

        if all_paths:
            args = params + ['-d', str(index_path)] + [str(t) for t in targets]
            _, stderr = self.run(args)
        else:
            # Stream records to stdin ('-')
            args = params + ['-d', str(index_path), '-']
            # Consume generator locally
            for _ in self._stream_input_output(args, targets, FastaWriter): pass

        if not is_non_empty_file(index_path):
            # Check for specific stderr info if run() captured it, otherwise vague error
            raise Minimap2Error(f'Failed to build index at {index_path}')

        self._target_index = index_path
        self._temp_files.append(index_path)
        return index_path

    def align(self, *queries: Record, targets: Iterable[Record] = (), config: Minimap2AlignConfig = None) -> Generator[Alignment, None, None]:
        if not config: config = self._align_config

        target_arg = None
        if targets:
            target_arg = str(self.build_index(*targets))
        elif self._target_index:
            target_arg = str(self._target_index)
        else:
            raise Minimap2Error('Targets must be supplied if no target index has been built')

        params = _build_params(config)
        args = params + [target_arg, '-']

        output_stream = self._stream_input_output(args, queries, FastaWriter)
        yield from PafReader(output_stream)


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
    _ALPHABET = Alphabet.amino()
    _DEFAULT_CONFIG = FragGeneScanRsConfig()

    def __init__(self, config: FragGeneScanRsConfig = None):
        super().__init__('FragGeneScanRs')
        self._config = config or self._DEFAULT_CONFIG

    def predict(self, *seqs: Record, config: FragGeneScanRsConfig = None) -> Generator[Record, None, None]:
        config = config or self._config
        args = _build_params(config) + ['--seq-file-name', 'stdin']

        seq_map = {i.id: i for i in seqs}

        stdout_lines = self._stream_input_output(args, seqs, FastaWriter)
        reader = FastaReader(stdout_lines, alphabet=self._ALPHABET)

        for record in reader:
            try:
                # FragGeneScanRs header: >parent_start_end_strand
                parent_id, start, end, strand = record.id.rsplit('_', 3)
                if parent_id in seq_map:
                    seq_map[parent_id].features.append(
                        Feature(
                            Interval(int(start) - 1, int(end), strand),
                            kind='CDS',
                            qualifiers=[
                                Qualifier('transl_table', 11),
                                Qualifier('inference', f'ab initio prediction:FragGeneScanRs:{self.version()}'),
                            ]
                        )
                    )
            except ValueError:
                # Robustness: Don't crash if FGS outputs a weird header, just skip associating the feature
                pass

            yield record


# Helpers --------------------------------------------------------------------------------------------------------------
def _build_params(config: Config) -> list[str]:
    params = []
    for k, v in asdict(config).items():
        if v is None or v is False: continue
        flag = f"-{k}" if len(k) == 1 else f"--{k.replace('_', '-')}"
        if v is True: params.append(flag)
        else:
            params.append(flag)
            params.append(str(v))
    return params
