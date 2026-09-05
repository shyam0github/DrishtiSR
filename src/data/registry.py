"""Dataset registry: name -> :class:`SRPairDataset` subclass.

Training, evaluation, and export code calls :func:`get_dataset` and nothing else.
Switching from the primary dataset to a fallback is an edit to
``cfg.dataset.name``; adding a new dataset is a new module plus one
``@register_dataset`` line. Neither touches training code.

Registration is lazy. Concrete datasets pull in heavy, optional dependencies
(``tacoreader``, ``rasterio``), so their modules are imported only when the
dataset is actually requested. That is what lets ``--smoke`` run on a bare CPU
box with neither installed.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from src.data.base import SRPairDataset

__all__ = [
    "register_dataset",
    "get_dataset",
    "get_dataset_class",
    "available_datasets",
]

# Datasets shipped in this repo, as name -> (module, class name). Listing them
# here rather than importing at module load keeps optional dependencies optional.
_LAZY_DATASETS: Dict[str, Tuple[str, str]] = {
    "sen2naipv2": ("src.data.sen2naip", "SEN2NAIPv2Dataset"),
    "worldstrat": ("src.data.worldstrat", "WorldStratDataset"),
    "synthetic_stub": ("src.data.synthetic", "SyntheticStubDataset"),
}

# Filled by @register_dataset as modules are imported.
_REGISTRY: Dict[str, Type[SRPairDataset]] = {}


def register_dataset(name: str) -> Callable[[Type[SRPairDataset]], Type[SRPairDataset]]:
    """Class decorator registering a dataset under ``name``.

    Args:
        name: The value ``cfg.dataset.name`` must take to select this class.

    Raises:
        TypeError: The decorated class is not an :class:`SRPairDataset`.
        ValueError: ``name`` is already registered to a different class.
    """

    def decorator(cls: Type[SRPairDataset]) -> Type[SRPairDataset]:
        if not (isinstance(cls, type) and issubclass(cls, SRPairDataset)):
            raise TypeError(
                f"@register_dataset({name!r}) applied to {cls!r}, which is not a "
                "subclass of SRPairDataset. Every dataset must implement the "
                "interface so training code never special-cases a source."
            )
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Dataset name {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}; refusing to "
                f"rebind it to {cls.__module__}.{cls.__qualname__}."
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_datasets() -> List[str]:
    """All registered dataset names, including not-yet-imported built-ins."""
    return sorted(set(_LAZY_DATASETS) | set(_REGISTRY))


def get_dataset_class(name: str) -> Type[SRPairDataset]:
    """Resolve a dataset name to its class, importing its module if needed.

    Args:
        name: A key from :func:`available_datasets`.

    Returns:
        The :class:`SRPairDataset` subclass.

    Raises:
        KeyError: ``name`` is not registered. The message lists what is.
        ImportError: The module exists but its dependencies are missing. Raised
            with the underlying cause chained, so a missing ``rasterio`` reads
            as a missing ``rasterio`` and not as an unknown dataset.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]

    if name not in _LAZY_DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered datasets: "
            f"{available_datasets()}. Set cfg.dataset.name to one of these, or "
            "add a module with @register_dataset."
        )

    module_name, class_name = _LAZY_DATASETS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Dataset {name!r} lives in {module_name}, which failed to import: "
            f"{exc}. This is usually a missing optional dependency -- see "
            "requirements.txt. Install it, or select a different "
            "cfg.dataset.name."
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"{module_name} does not define {class_name!r}, which the registry "
            f"expects for dataset {name!r}."
        ) from exc

    # Importing the module should have run @register_dataset. If the module
    # forgot the decorator, bind it here so behaviour stays predictable, but
    # only after the type check that the decorator would have done.
    if not (isinstance(cls, type) and issubclass(cls, SRPairDataset)):
        raise TypeError(
            f"{module_name}.{class_name} is not an SRPairDataset subclass."
        )
    _REGISTRY.setdefault(name, cls)
    return cls


def get_dataset(
    cfg: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> SRPairDataset:
    """Construct the dataset selected by ``cfg.dataset.name``.

    This is the only dataset entry point training code may use.

    Args:
        cfg: The loaded config.
        name: Override ``cfg.dataset.name``. Intended for tooling that inspects
            several datasets in one process (e.g. the manifest builder); normal
            code passes only ``cfg``.
        **kwargs: Forwarded to the dataset constructor, e.g. ``validate=False``.

    Returns:
        A ready-to-use :class:`SRPairDataset`.

    Raises:
        KeyError: ``cfg.dataset.name`` is missing or unregistered.
    """
    if name is None:
        try:
            name = cfg["dataset"]["name"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "cfg.dataset.name is not set; it selects the dataset "
                f"implementation. Registered: {available_datasets()}."
            ) from exc

    return get_dataset_class(name)(cfg, **kwargs)
