# Contributing to baclib

This document describes the conventions and architectural patterns used throughout `baclib`. Please read it before contributing code.

## Project Structure

```
src/baclib/
├── containers/   # Data containers (Seq, Record, Feature, Alignment, etc.)
├── core/         # Core primitives (Alphabet, Interval)
├── engines/      # Compute engines (Aligner, Index, Motif Scanner)
├── io/           # File readers and writers (FASTA, GenBank, GFF, PAF, etc.)
├── apis/         # REST API clients (PRODORIC, Datasets)
└── lib/          # Shared utilities (Resources, Protocols, External Programs, CLI)
```

**Dependency direction**: `lib` → `containers` → `core` → `engines` / `io` / `apis`. Higher-level modules import from lower-level ones, never the reverse. Circular imports between `containers` and `core` are avoided by deferring imports (e.g. `from baclib.core.alphabet import Alphabet` inside method bodies).

---

## Core Paradigms

### 1. Batch / Batchable System

Every domain object has a **scalar** form and a **batch** form:

| Scalar (`Batchable`) | Batch (`Batch`) |
|---|---|
| `Interval` | `IntervalBatch` |
| `Seq` | `SeqBatch` |
| `Feature` | `FeatureBatch` |
| `Record` | `RecordBatch` |
| `Alignment` | `AlignmentBatch` |

**Rules:**
- Scalars implement `Batchable` (an ABC with a `batch` property returning the batch type).
- Batches implement `Batch` (an ABC enforcing `__len__`, `__getitem__`, `empty()`, `build()`, `concat()`, and a `component` property returning the scalar type).
- `RaggedBatch` extends `Batch` for variable-length items using a CSR-like (offsets + flat data) layout.
- Batch constructors should accept raw arrays for zero-overhead internal construction. User-facing creation should go through factory classmethods (`build`, `zeros`, `random`, `empty`).

### 2. Alphabet-Owned Sequence Creation

`Seq`, `SeqBatch`, `CompressedSeq`, and `CompressedSeqBatch` **must** be created via `Alphabet` factory methods. Direct construction raises `PermissionError`.

```python
# ✅ Correct
seq = Alphabet.DNA.seq_from(b"ATCG")
batch = Alphabet.DNA.random_batch(n_seqs=10)
compressed = Alphabet.DNA.compress(seq)

# ❌ Wrong — raises PermissionError
seq = Seq(data, alphabet)
```

This is enforced with a `_validation_token` parameter that must match the `alphabet` argument. The pattern ensures encoding/decoding consistency and prevents data from drifting from its metadata.

**Alphabet factory methods:**

| Purpose | Seq | SeqBatch | CompressedSeq | CompressedSeqBatch |
|---|---|---|---|---|
| From data | `seq_from()` | `batch_from()` | `compress()` | `compress()` |
| Empty | `empty_seq()` | `empty_batch()` | — | `empty_compressed()` |
| Random | `random_seq()` | `random_batch()` | — | — |
| Raw (internal) | `new_seq()` | `new_batch()` | `new_compressed_seq()` | `new_compressed_batch()` |

### 3. Structure-of-Arrays (SoA) Layout

Batches store data in flat, contiguous NumPy arrays rather than lists of objects. This enables:
- Numba `@jit` parallel processing (`prange`)
- Cache-friendly memory access
- Zero-copy slicing via NumPy views

```python
# SeqBatch stores:
#   _data:    [A C G T | T A C | G G G A T]   ← flat encoded bytes
#   _starts:  [0, 4, 7]                        ← start offsets into _data
#   _lengths: [4, 3, 5]                        ← length of each sequence

### 4. Bytes DType for Strings

Batches **must not use** object-dtype arrays for strings (identifiers, keys, etc.). Use NumPy's 'S' (bytes) dtype instead:

```python
# ✅ Correct
self._ids = np.array([b"seq1", b"seq2"], dtype='S')

# ❌ Wrong
self._ids = np.array(["seq1", "seq2"], dtype=object)
```

**Handling Missing Values:**
- `None` strings should be encoded as empty bytes (`b""`).
- If distiguishing between `None` and `""` is critical, use a separate validity mask (boolean array).

### 5. Protocols (Structural Typing)
```

### 4. Protocols (Structural Typing)

Use `@runtime_checkable` protocols from `lib/protocols.py` for duck-typed interfaces:

- `HasAlphabet` — objects with an `.alphabet` property
- `HasInterval` — objects with an `.interval` property
- `HasIntervals` — objects with an `.intervals` property

Prefer these over inheritance when a class needs to advertise a capability without inheriting from a specific batch type.

---

## Code Style

### `__slots__`

All data classes use `__slots__` to minimise memory and prevent accidental attribute creation. Every class that stores data should define `__slots__`.

```python
class Feature:
    __slots__ = ('_interval', '_key', '_qualifiers')
```

### Immutable Data

Arrays backing `Seq` and `SeqBatch` are made immutable after construction:

```python
self._data.flags.writeable = False
```

This ensures hash safety and prevents silent mutation of shared views.

### Section Comments

Files are organised with horizontal-rule section comments:

```python
# Classes -----------------------------------------------------------------------
# Kernels -----------------------------------------------------------------------
# Constants ---------------------------------------------------------------------
# Exceptions and Warnings -------------------------------------------------------
# Drivers -----------------------------------------------------------------------
```

### Enumerations

Use `IntEnum` for values that must participate in NumPy arrays or Numba kernels (e.g. `Strand`, `CigarOp`, `AlignmentMode`). Use `Enum` for symbolic-only types.

### Docstrings

Use Google-style docstrings with `Args:`, `Returns:`, `Raises:`, and `Examples:` sections.

---

## Performance Patterns

### Conditional Numba JIT

Import and use the `@jit` decorator from `lib/resources.py`, **not** directly from Numba:

```python
from baclib.lib.resources import jit

@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _my_kernel(data, starts, lengths, out):
    for i in prange(len(starts)):
        ...
```

If Numba is not installed, `@jit` is a no-op passthrough — the function runs as plain Python. Always guard `from numba import prange` with:

```python
if RESOURCES.has_module('numba'):
    from numba import prange
else:
    prange = range
```

### Kernel / Driver Separation

Numba kernels are pure numeric functions (no Python objects, no allocations). They live at module level, prefixed with `_` and suffixed with `_kernel`.

A **driver** function (Python-level) handles:
1. Preparing input arrays
2. Allocating output arrays
3. Calling the kernel
4. Wrapping results in domain objects

```python
# Driver (Python)
def intersect(self, other):
    out = _intersect_kernel(self.starts, self.ends, ...)
    return IntervalBatch(out[0], out[1], out[2])

# Kernel (Numba)
@jit(nopython=True, cache=True)
def _intersect_kernel(a_starts, a_ends, b_starts, b_ends, ...):
    ...
```

### Zero-Copy Optimisations

Prefer NumPy views and slices over copies. When slicing a batch, check for physical contiguity before falling through to a gather:

```python
if (s_end - s_start) == new_lengths.sum():
    # Contiguous — zero-copy slice
    new_data = self._data[s_start:s_end]
else:
    # Non-contiguous — must gather
    ...
```

---

## I/O Conventions

### Readers

Subclass `BaseReader` from `io/__init__.py`. Key requirements:
- Implement `_parse()` as a generator yielding domain objects.
- Implement `sniff(cls, s: bytes)` as a classmethod to detect file format from header bytes.
- Implement `_make_batch(self, records)` for efficient batched reading.
- All readers operate on **binary** file handles (`BinaryIO`), never text mode.

### Writers

Subclass `BaseWriter`. Key requirements:
- Implement `write_one(self, item)` for single-item writing.
- Optionally override `write_batch(self, batch)` for batch-optimised output.
- Writers support context managers and automatic compression detection.

---

## External Tools

Wrap external command-line programs by subclassing `ExternalProgram` from `lib/external.py`:

- Use `@dataclass` config objects (e.g. `Minimap2AlignConfig`) for CLI parameter management.
- Use `_stream_input_output()` for piping `Record` objects to/from subprocesses.
- Validate binary availability via `RESOURCES.find_binary()`.

---

## API Clients

Subclass `ApiClient` from `apis/__init__.py`:

- Built on `urllib` (no `requests` dependency).
- Rate limiting, retries, and exponential backoff are handled by the base class.
- Implement domain-specific methods that call `self.get()` / `self.post()` and parse responses into `baclib` containers.

---

## Global Resources

Use the singleton `RESOURCES` from `lib/resources.py` for:

| Resource | Access |
|---|---|
| Random number generator | `RESOURCES.rng` |
| Thread pool | `RESOURCES.pool` |
| CPU count | `RESOURCES.available_cpus` |
| Module availability | `RESOURCES.has_module('numba')` |
| Binary lookup | `RESOURCES.find_binary('minimap2')` |

All properties are lazily initialised via `@cached_property`.
