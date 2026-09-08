"""The sample-cache manifest: what is on disk, recorded rather than re-derived.

Why a manifest and not a directory glob
---------------------------------------
The cache is a flat directory of ``<taco_id>.npz`` / ``<taco_id>.json`` pairs. A
glob over it answers "which files exist", which is not the question. The
question is "which PAIRS are complete, validated, and safe to train on", and a
filename cannot answer that:

- A glob counts a ``.npz`` that was truncated by an interrupted download.
- A glob cannot tell a 130/520 pair from a 130/260 one without opening it, and
  opening 3000 compressed NPZ files to find out costs minutes on every startup.
- A glob has no memory of *why* a pair was rejected, so a rejected download is
  re-attempted on every run, forever.

So the manifest is the record of record. A line exists in it only after the pair
was written, re-read, and validated; :func:`read_manifest` is therefore the
authoritative "is this cached" answer, and the ``.npz`` on disk is merely the
payload it points at.

Format
------
JSONL, one object per cached pair, appended and never rewritten in place:

.. code-block:: json

    {"taco_id": "NA5120_...", "lr_path": "NA5120_....npz", "hr_path": "...",
     "lr_shape": [4, 130, 130], "hr_shape": [4, 520, 520],
     "bands": ["B04", "B03", "B02", "B08"], "cached_at": "2026-09-09T12:00:00Z",
     "dtype": "uint16", "nodata_value": 65535, "reflectance_scale": 10000.0}

``lr_path`` and ``hr_path`` are RELATIVE to the manifest's directory, so the
cache stays portable between ``D:\\SIH\\DrishtiSR\\outputs\\cache`` and
``/kaggle/input/drishtisr-sen2naipv2-cache``. They are equal for SEN2NAIPv2,
where both halves live in one NPZ under keys ``lr`` and ``hr``; the two columns
exist because a dataset that stores them separately must be describable without
a second manifest schema.

Append-only, and the one tolerated corruption
---------------------------------------------
Records are appended with an ``fsync`` so a line is either fully on disk or not
started. A process killed mid-append can still leave a torn final line. That one
case is tolerated explicitly: the last line, if unparseable, is dropped with a
WARNING and the pair is re-validated on the next run. An unparseable line
anywhere ELSE is a corrupt manifest and raises -- it means something rewrote the
file, and silently skipping those rows would quietly shrink the training set.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MANIFEST_NAME",
    "UNREADABLE_CACHE_ERRORS",
    "CacheValidationError",
    "manifest_path",
    "read_manifest",
    "append_record",
    "make_record",
    "validate_pair",
    "npz_member_shape",
    "REJECTION_REASONS",
]

# Lives at the cache root, beside the per-subset directories' contents.
MANIFEST_NAME = "cache_manifest.jsonl"

# What reading a damaged cache entry actually raises. Collected here because
# getting it wrong is a live bug rather than a hypothetical one: an NPZ is a zip
# archive, and a truncated one -- the signature of an interrupted download, the
# single most likely damage in this cache -- raises ``zipfile.BadZipFile``,
# which is a plain ``Exception`` and NOT an ``OSError``. A handler catching only
# (OSError, ValueError, EOFError) therefore lets it through and crashes the run
# it was written to protect. Caught by tests/test_cache_expand.py.
UNREADABLE_CACHE_ERRORS = (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile)

# Every reason a pair can be refused. Kept as a tuple so the tally printed at the
# end of a run has a fixed set of keys rather than whichever ones happened to
# fire -- a reason with a count of zero is information too.
REJECTION_REASONS = (
    "fetch_failed",
    "scale_mismatch",
    "band_count_mismatch",
    "dtype_mismatch",
    "all_zero",
    "all_nan",
    "dn_range_implausible",
    "shape_degenerate",
)


class CacheValidationError(Exception):
    """A fetched pair failed validation and must not be committed to the cache.

    Carries ``reason``, one of :data:`REJECTION_REASONS`, so the caller can tally
    rejections by cause instead of by exception text.
    """

    def __init__(self, reason: str, message: str) -> None:
        if reason not in REJECTION_REASONS:
            raise ValueError(
                f"Unknown rejection reason {reason!r}. Add it to "
                f"REJECTION_REASONS; the tally is keyed on that tuple so an "
                f"unlisted reason would be counted under nothing."
            )
        super().__init__(message)
        self.reason = reason
        self.message = message


def manifest_path(cache_dir: Any) -> Path:
    """Path to the manifest for a cache directory.

    Args:
        cache_dir: The directory holding the cached ``.npz``/``.json`` pairs,
            i.e. what ``resolve_cache_dir(cfg) / subset`` returns.

    Returns:
        ``<cache_dir>/cache_manifest.jsonl``. Not created; may not exist.
    """
    return Path(cache_dir) / MANIFEST_NAME


def read_manifest(path: Any) -> Dict[str, Dict[str, Any]]:
    """Read a JSONL manifest into ``taco_id -> record``.

    Args:
        path: The manifest file. A missing file is an empty cache, not an error.

    Returns:
        Mapping of taco id to its record, in file order (dicts preserve it). A
        taco id appearing twice keeps the LAST record, so a re-validated pair
        supersedes its earlier line without the file needing a rewrite.

    Raises:
        ValueError: A line other than the last one is not valid JSON, or a
            record is missing ``taco_id``. Both mean the file was rewritten by
            something that does not understand the format; skipping those rows
            would silently shrink the training set.
    """
    path = Path(path)
    if not path.is_file():
        return {}

    records: Dict[str, Dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines):
                # The one tolerated corruption: a process killed mid-append. The
                # pair it describes is simply re-validated on the next run.
                import logging

                logging.getLogger("data.cache_manifest").warning(
                    "Dropping a torn final line in %s (offset %d): %s. It is the "
                    "signature of a run killed mid-append; the pair it described "
                    "will be re-validated.",
                    path,
                    number,
                    exc,
                )
                continue
            raise ValueError(
                f"{path}:{number} is not valid JSON ({exc}). Only the FINAL "
                "line may be torn (an interrupted append); a bad line in the "
                "middle means the manifest was rewritten by something else. "
                "Delete it and re-run the cache expansion to rebuild it from "
                "the pairs on disk."
            ) from exc

        taco_id = record.get("taco_id")
        if not taco_id:
            raise ValueError(
                f"{path}:{number} has no 'taco_id'. Every record must name the "
                f"pair it describes. Got keys {sorted(record)}."
            )
        records[str(taco_id)] = record

    return records


def append_record(path: Any, record: Mapping[str, Any]) -> None:
    """Append one record to the manifest and flush it to disk.

    The ``fsync`` is the point: this call is the commit that makes a pair
    "cached", so it must survive the process dying immediately afterwards. A
    buffered write that is lost on a kill would leave the ``.npz`` on disk and
    unrecorded, and the next run would pay to re-download it.

    Args:
        path: The manifest file. Its parent must exist.
        record: The record, as built by :func:`make_record`.
    """
    line = json.dumps(record, sort_keys=True) + "\n"
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def make_record(
    taco_id: str,
    lr_path: str,
    hr_path: str,
    lr_shape: Sequence[int],
    hr_shape: Sequence[int],
    bands: Sequence[str],
    dtype: str,
    nodata_value: Any,
    reflectance_scale: float,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a manifest record.

    Args:
        taco_id: The catalog's ``tortilla:id``, the pair's identity.
        lr_path: Path to the file holding the LR array, RELATIVE to the manifest
            directory.
        hr_path: Likewise for HR. Equal to ``lr_path`` when one NPZ holds both.
        lr_shape: ``(C, H, W)`` of the stored LR array, channels first.
        hr_shape: ``(C, H*scale, W*scale)`` of the stored HR array.
        bands: Band names in stored channel order, e.g.
            ``("B04", "B03", "B02", "B08")``.
        dtype: numpy dtype name of the STORED arrays, e.g. ``"uint16"``. The
            cache holds raw digital numbers, not reflectance.
        nodata_value: The DN meaning "no observation", e.g. 65535.
        reflectance_scale: The divisor that turns a stored DN into surface
            reflectance. Recorded per pair so a cache written under a different
            divisor is detectable rather than silently mixed in.
        **extra: Additional measured fields, e.g. ``lr_dn_p999``.

    Returns:
        A JSON-serialisable dict. ``cached_at`` is UTC ISO-8601 with a ``Z``.
    """
    record = {
        "taco_id": str(taco_id),
        "lr_path": str(lr_path),
        "hr_path": str(hr_path),
        "lr_shape": [int(d) for d in lr_shape],
        "hr_shape": [int(d) for d in hr_shape],
        "bands": [str(b) for b in bands],
        "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dtype": str(dtype),
        "nodata_value": None if nodata_value is None else int(nodata_value),
        "reflectance_scale": float(reflectance_scale),
    }
    record.update(extra)
    return record


def npz_member_shape(npz_path: Any, member: str) -> Tuple[Tuple[int, ...], str]:
    """Read one NPZ member's shape and dtype WITHOUT decompressing its data.

    An ``.npy`` inside an ``.npz`` starts with a header that states shape and
    dtype, so both are recoverable from the first few hundred bytes of the zip
    member's decompression stream, without inflating the array behind it.

    MEASURED 2026-09-09 over 50 pairs of the local cache, warm: 17.8 ms/file for
    both members by header, against 28.2 ms/file via ``np.load``. Over the
    3002-pair cache that is 53 s against 85 s. A 1.6x saving on a one-time
    backfill, not the order of magnitude the trick sounds like -- most of the
    cost is opening the zip, not inflating it. It is used because it is also the
    correct read: it asks the file what shape it holds instead of decompressing
    a megabyte to find out.

    Args:
        npz_path: Path to the ``.npz``.
        member: Array name inside it, e.g. ``"lr"``.

    Returns:
        ``(shape, dtype_name)`` where shape is ``(C, H, W)`` for this cache's
        arrays and ``dtype_name`` is e.g. ``"uint16"``.

    Raises:
        KeyError: The member is not in the archive.
        ValueError: The member is not a well-formed ``.npy`` stream.
    """
    with zipfile.ZipFile(npz_path) as archive:
        name = f"{member}.npy"
        if name not in archive.namelist():
            raise KeyError(
                f"{npz_path} has no member {name!r}; it holds "
                f"{archive.namelist()}."
            )
        with archive.open(name) as stream:
            major, minor = np.lib.format.read_magic(stream)
            # The public per-version readers, not the private dispatcher
            # np.lib.format._read_array_header, which is not API and has moved
            # between numpy releases.
            readers = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }
            reader = readers.get((major, minor))
            if reader is None:
                raise ValueError(
                    f"{npz_path}:{name} declares .npy format {major}.{minor}, "
                    f"which this reader does not know. Known: "
                    f"{sorted(readers)}."
                )
            shape, _fortran, dtype = reader(stream)
    return tuple(int(d) for d in shape), str(np.dtype(dtype).name)


def validate_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    *,
    scale: int,
    expected_bands: int,
    expected_dtype: str,
    nodata_value: Optional[int],
    dn_plausible_range: Tuple[float, float],
    taco_id: str = "<unknown>",
) -> Dict[str, Any]:
    """Check one fetched pair against the invariants the cache guarantees.

    Called BEFORE the pair is committed, so a failure costs one download rather
    than a corrupted training set. Every check states the measured value in its
    message, because "rejected: scale_mismatch" without the shapes is not
    actionable.

    Args:
        lr: Stored LR array, ``(C, H, W)``, raw ``uint16`` digital numbers --
            NOT reflectance. The cache is lossless DN so a corrected divisor
            does not invalidate the download.
        hr: Stored HR array, ``(C, H*scale, W*scale)``, same dtype and domain.
        scale: The super-resolution factor, ``cfg.sr.scale``. HR must be exactly
            this multiple of LR in BOTH spatial dimensions -- "about 4x" means a
            resampling step crept in somewhere and the pair is not co-registered.
        expected_bands: Channel count both arrays must have,
            ``len(NATIVE_BANDS)``.
        expected_dtype: numpy dtype name every cached pair shares, ``"uint16"``.
        nodata_value: DN meaning no observation, excluded from the all-zero and
            DN-range checks. None disables the exclusion.
        dn_plausible_range: ``(low, high)`` bounds on the 99.9th percentile of
            valid LR digital numbers, from
            ``cfg.sen2naipv2.cache_validation.dn_p999_range``. This is the
            reflectance-scaling check: a pair delivered under a different
            quantification value lands one or two orders of magnitude outside
            the band the existing cache occupies. It is a scaling check, not a
            brightness clip -- nothing is modified, the pair is refused whole.
        taco_id: For the messages.

    Returns:
        Measured statistics worth recording in the manifest:
        ``{"lr_shape", "hr_shape", "dtype", "lr_dn_p999", "nodata_fraction"}``.

    Raises:
        CacheValidationError: Any invariant failed. ``.reason`` is one of
            :data:`REJECTION_REASONS`.
    """
    if lr.ndim != 3 or hr.ndim != 3:
        raise CacheValidationError(
            "shape_degenerate",
            f"{taco_id}: expected (C, H, W) for both halves, got LR "
            f"{lr.shape} and HR {hr.shape}.",
        )
    if min(lr.shape) == 0 or min(hr.shape) == 0:
        raise CacheValidationError(
            "shape_degenerate",
            f"{taco_id}: a zero-length axis -- LR {lr.shape}, HR {hr.shape}.",
        )

    if lr.shape[0] != expected_bands or hr.shape[0] != expected_bands:
        raise CacheValidationError(
            "band_count_mismatch",
            f"{taco_id}: expected {expected_bands} bands, got LR "
            f"{lr.shape[0]} and HR {hr.shape[0]}. Every cached pair must carry "
            "the same channels in the same order or the band indices used at "
            "load time select the wrong wavelengths.",
        )

    if hr.shape[1] != lr.shape[1] * scale or hr.shape[2] != lr.shape[2] * scale:
        raise CacheValidationError(
            "scale_mismatch",
            f"{taco_id}: HR {hr.shape[1]}x{hr.shape[2]} is not exactly "
            f"{scale}x LR {lr.shape[1]}x{lr.shape[2]} (expected "
            f"{lr.shape[1] * scale}x{lr.shape[2] * scale}).",
        )

    if str(lr.dtype) != expected_dtype or str(hr.dtype) != expected_dtype:
        raise CacheValidationError(
            "dtype_mismatch",
            f"{taco_id}: expected {expected_dtype} on disk, got LR "
            f"{lr.dtype} and HR {hr.dtype}. The cache stores raw digital "
            "numbers; a float array here means something already applied the "
            "reflectance divisor and the pair would be divided twice.",
        )

    # NaN cannot occur in an integer array, so this only ever fires if the
    # source hands back floats. Checked anyway -- it is the cheap half of "no
    # all-NaN tiles" and stays correct if the stored dtype ever changes.
    for name, array in (("LR", lr), ("HR", hr)):
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).any():
            raise CacheValidationError(
                "all_nan",
                f"{taco_id}: every {name} value is NaN or infinite.",
            )

    if nodata_value is None:
        lr_valid = np.ones(lr.shape[1:], dtype=bool)
        hr_valid = np.ones(hr.shape[1:], dtype=bool)
    else:
        # A pixel is valid only where NO band is nodata: one dead band makes the
        # whole pixel unusable, matching load_sample's ANY-band rule.
        lr_valid = ~np.any(lr == np.asarray(nodata_value, dtype=lr.dtype), axis=0)
        hr_valid = ~np.any(hr == np.asarray(nodata_value, dtype=hr.dtype), axis=0)

    if not lr_valid.any() or not hr_valid.any():
        raise CacheValidationError(
            "all_nan",
            f"{taco_id}: no valid pixels -- LR has {int(lr_valid.sum())} and HR "
            f"{int(hr_valid.sum())} pixels where every band is an observation. "
            "The tile is entirely nodata.",
        )

    lr_values = lr[:, lr_valid]
    hr_values = hr[:, hr_valid]
    if not lr_values.any() or not hr_values.any():
        raise CacheValidationError(
            "all_zero",
            f"{taco_id}: every valid pixel is 0 in one half (LR max "
            f"{int(lr_values.max())}, HR max {int(hr_values.max())}). A "
            "uniformly black tile teaches the model nothing and inflates any "
            "metric averaged over it.",
        )

    lr_p999 = float(np.percentile(lr_values.astype(np.float64), 99.9))
    low, high = float(dn_plausible_range[0]), float(dn_plausible_range[1])
    if not (low <= lr_p999 <= high):
        raise CacheValidationError(
            "dn_range_implausible",
            f"{taco_id}: the 99.9th percentile of valid LR digital numbers is "
            f"{lr_p999:.0f}, outside the [{low:.0f}, {high:.0f}] band the "
            "existing cache occupies. Either the pair was delivered under a "
            "different quantification value -- in which case dividing it by "
            "cfg.dataset.reflectance_scale gives wrong reflectance -- or it is "
            "a degenerate tile. Refused whole; nothing is clipped.",
        )

    nodata_fraction = max(
        float((~lr_valid).mean()), float((~hr_valid).mean())
    )

    return {
        "lr_shape": [int(d) for d in lr.shape],
        "hr_shape": [int(d) for d in hr.shape],
        "dtype": str(lr.dtype),
        "lr_dn_p999": lr_p999,
        "nodata_fraction": nodata_fraction,
    }


def format_rejection_tally(rejects: Mapping[str, int]) -> List[str]:
    """Render a rejects-by-reason tally, listing zero counts too.

    A reason that fired zero times is evidence the check ran and passed, so it
    is printed rather than omitted.

    Args:
        rejects: Reason -> count. Reasons outside :data:`REJECTION_REASONS` are
            appended after the known ones rather than dropped.

    Returns:
        Lines, ready to join with newlines.
    """
    known = [f"  {reason:<22} {rejects.get(reason, 0)}" for reason in REJECTION_REASONS]
    unknown = [
        f"  {reason:<22} {count}  (UNLISTED REASON)"
        for reason, count in sorted(rejects.items())
        if reason not in REJECTION_REASONS
    ]
    return known + unknown
