try: from ._version import __version__
except ImportError: __version__ = "unknown"  # Fallback for when running directly from source without installation

from .core.seq import Seq, Alphabet
from .io.dispatcher import SeqFile, Xopen
from .containers.record import Record, Feature
from .containers.graph import Graph
from .containers.genome import Genome

# Lazy module getattr (Python 3.7+)
def __getattr__(name):
    if name == 'io':
        import baclib.io as _io
        return _io
    if name == 'align':
        import baclib.align.alignment as _align
        return _align
    raise AttributeError(f"module {__name__} has no attribute {name}")
