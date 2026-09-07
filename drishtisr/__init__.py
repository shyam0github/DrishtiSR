"""Import alias: ``drishtisr.X`` resolves to the module ``src/X``.

WHY THIS EXISTS. The package root of this repository is ``src/`` -- Day 1 code
imports ``src.data.sen2naip``, ``src.utils.paths`` and so on, and there is no
``pip install`` step anywhere in the workflow: every entry point runs with the
repository root as the working directory, so ``src`` is importable because the
root is on ``sys.path``.

The Day 2 files (``src/models/edsr.py``, ``src/infer/tiled.py``,
``src/train.py``) were written against a ``drishtisr.*`` package root instead.
Rather than edit three finished files, or restructure the Day 1 tree around a
name it never used, this package makes both names resolve to the same
directory by pointing its ``__path__`` at ``src/``.

Consequence, stated rather than hidden: a module imported under BOTH names is
two distinct objects (``src.models.edsr`` is not ``drishtisr.models.edsr``).
That is harmless for the stateless modules involved here -- no module-level
registry, cache, or singleton is shared across the two -- but it is a real trap
if module-level state is ever added. The registry that DOES hold state,
``src.data.registry``, is only ever reached through the ``src.`` name.

The mechanism is identical on Kaggle: the kernel clones the repository into
``/kaggle/working/DrishtiSR`` and runs with that as the working directory, so
this package is found the same way ``src`` is. Nothing is installed.
"""
from pathlib import Path

# Submodules of `drishtisr` are searched for in `<repo root>/src`.
__path__ = [str(Path(__file__).resolve().parent.parent / "src")]
