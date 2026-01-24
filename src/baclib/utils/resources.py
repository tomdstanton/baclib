"""
Top-level module, including resource and optional dependency management.
"""
from functools import cached_property
from importlib import import_module
# from importlib.metadata import metadata as load_metadata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from numpy.random import default_rng
import atexit
import os
from typing import Callable


# Classes --------------------------------------------------------------------------------------------------------------
class Resources:
    def __init__(self, *optional_packages: str):
        self.package = Path(__file__).parent.name
        self.optional_packages = set(filter(self._check_module, optional_packages))
        # Register cleanup to run automatically when the program exits
        atexit.register(self.shutdown)

    # @cached_property
    # def data(self): return resources.files(self.package) / 'data'

    # @cached_property
    # def metadata(self): return load_metadata(self.package)

    @cached_property
    def rng(self): return default_rng()

    @cached_property
    def available_cpus(self) -> int:
        try: return os.process_cpu_count()
        except AttributeError: return os.cpu_count()

    @cached_property
    def pool(self) -> ThreadPoolExecutor: return ThreadPoolExecutor(min(32, (self.available_cpus or 1) + 4))

    def shutdown(self):
        # Check if 'pool' is in __dict__ (meaning it was initialized)
        if 'pool' in self.__dict__: self.pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _check_module(module_name: str) -> bool:
        try:
            import_module(module_name)
            return True
        except ImportError: return False

    # __enter__ and __exit__ are still useful for scoped usage (e.g. testing)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.shutdown()


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
    if 'numba' not in RESOURCES.optional_packages:
        if callable(signature_or_function): return signature_or_function  # Handle bare @jit
        def passthrough(func: Callable) -> Callable: return func  # Handle @jit(...)
        return passthrough
    # 2. Apply Numba
    from numba import jit as real_jit
    if callable(signature_or_function): return real_jit(signature_or_function)  # Handle bare @jit
    return real_jit(signature_or_function, **options)  # Handle @jit(...)


# Constants ------------------------------------------------------------------------------------------------------------
RESOURCES = Resources('numba')

