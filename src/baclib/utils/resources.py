"""
Top-level module, including resource and optional dependency management.
"""
from functools import cached_property, lru_cache
from importlib import import_module
from shutil import which
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from numpy.random import default_rng
import atexit
import os
from typing import Callable, Optional


# Classes --------------------------------------------------------------------------------------------------------------
class Resources:
    """
    Manages global resources like thread pools, random number generators,
    and optional dependencies.

    Attributes:
        package (str): The package name.
    """
    def __init__(self) -> None:
        self.package = Path(__file__).parent.name
        # Register cleanup to run automatically when the program exits
        atexit.register(self._cleanup)

    @cached_property
    def rng(self):
        """Returns a default numpy random number generator."""
        return default_rng()

    @cached_property
    def available_cpus(self) -> int:
        """Returns the number of available CPUs."""
        try: return os.process_cpu_count()
        except AttributeError: return os.cpu_count()

    @cached_property
    def pool(self) -> ThreadPoolExecutor:
        """Returns a shared ThreadPoolExecutor."""
        return ThreadPoolExecutor(min(32, (self.available_cpus or 1) + 4))

    @cached_property
    def has_gpu(self) -> bool:
        """Checks if a CUDA-compatible GPU is available via Numba."""
        if not self.has_module('numba'): return False
        try:
            from numba import cuda
            return cuda.is_available()
        except Exception: return False

    def _cleanup(self):
        """Shuts down the thread pool."""
        # Check if 'pool' is in __dict__ (meaning it was initialized)
        if 'pool' in self.__dict__: self.pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    @lru_cache(maxsize=None)
    def has_module(module_name: str) -> bool:
        """Checks if a python package is installed."""
        try:
            import_module(module_name)
            return True
        except ImportError: return False
        
    @staticmethod
    @lru_cache(maxsize=None)
    def find_binary(program_name: str) -> Optional[Path]:
        """
        Locates an executable in the system PATH.
        Returns the Path object if found, else None.
        """
        if path := which(program_name): return Path(path)
        return None

    # __enter__ and __exit__ are still useful for scoped usage (e.g. testing)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self._cleanup()


# Decorators -----------------------------------------------------------------------------------------------------------
def jit(signature_or_function=None, **options) -> Callable:
    """
    Conditional Numba JIT decorator.

    If 'numba' is installed (checked via RESOURCES), this applies `numba.jit`
    with the provided arguments. Otherwise, it returns the original function unmodified,
    ignoring any compilation options.

    Examples:
        >>> @jit  # Bare usage
        ... def func(): ...

        >>> @jit(nopython=True, cache=True)  # Configured usage
        ... def func(): ...
    """
    # 1. Fallback: Numba not installed
    if not RESOURCES.has_module('numba'):
        if callable(signature_or_function): return signature_or_function  # Handle bare @jit
        def passthrough(func: Callable) -> Callable: return func  # Handle @jit(...)
        return passthrough
    # 2. Apply Numba
    from numba import jit as real_jit
    if callable(signature_or_function): return real_jit(signature_or_function)  # Handle bare @jit
    return real_jit(signature_or_function, **options)  # Handle @jit(...)


# Constants ------------------------------------------------------------------------------------------------------------
RESOURCES = Resources()
