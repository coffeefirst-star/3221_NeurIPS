from bblean._config import DEFAULTS
from bblean.fingerprints import (
    _get_generator,
    _get_sanitize_flags,
    pack_fingerprints,
)
from rdkit.Chem import SDMolSupplier, SanitizeMol
import typing as tp
from pathlib import Path
import numpy as np
from numpy.typing import NDArray, DTypeLike

def fps_from_sdfs(
    sdf_paths: tp.Iterable[str | Path],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: bool = False,
    pack: bool = True,
) -> tp.Union[NDArray[np.uint8], tuple[NDArray[np.uint8], NDArray[np.int64]]]:
    """Convert SDF(s) into chemical fingerprints using BBLean utilities."""
    if n_features < 1 or n_features % 8 != 0:
        raise ValueError("n_features must be a multiple of 8, and greater than 0")
    if isinstance(sdf_paths, (str, Path)):
        sdf_paths = [sdf_paths]
    if pack and not (np.dtype(dtype) == np.dtype(np.uint8)):
        raise ValueError("Packing only supported for uint8 dtype")
    fpg = _get_generator(kind, n_features)
    sanitize_flags = _get_sanitize_flags(sanitize)
    mols = []
    for sdf_path in sdf_paths:
        suppl = SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        mols.extend(list(suppl))
    fps = np.empty((len(mols), n_features), dtype=dtype)
    invalid_idxs = []
    for i, mol in enumerate(mols):
        if mol is None:
            if skip_invalid:
                invalid_idxs.append(i)
                continue
            else:
                raise ValueError(f"Unable to parse molecule at idx {i} (None)")
        try:
            SanitizeMol(mol, sanitizeOps=sanitize_flags)
            fps[i, :] = fpg.GetFingerprintAsNumPy(mol)
        except Exception:
            if skip_invalid:
                invalid_idxs.append(i)
                continue
            raise
    if invalid_idxs:
        fps = np.delete(fps, invalid_idxs, axis=0)
    if pack:
        if skip_invalid:
            return pack_fingerprints(fps), np.array(invalid_idxs, dtype=np.int64)
        return pack_fingerprints(fps)
    if skip_invalid:
        return fps, np.array(invalid_idxs, dtype=np.int64)
    return fps


fps, _ = fps_from_sdfs(
    ["Enamine_4.5.sdf"],
    kind="ecfp4",
    n_features=2048,
    pack=True,
    skip_invalid=True,
)

np.save("fingerprints.npy", fps)
