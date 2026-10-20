"""Checkpoint cache for resuming a killed TB2J exchange calculation.

TB2J's energy-contour integral (the dominant cost of siesta2J.py and
friends) is a sum of independent per-energy-point contributions (one per
`--nz` contour point). Each point's contribution does not depend on any
other point, so if a run is killed partway through, the points already
computed can be reused instead of recomputed on the next run -- this is
the only safe checkpoint granularity available: there is no way to resume
*within* a single point's calculation.

Used by ExchangeCL2.calculate_all() when `resume=True` (--resume on
siesta2J.py and friends). Not wired into the non-collinear/GPU exchange
paths.
"""
import hashlib
import json
import os
import pickle
import shutil

CACHE_SUBDIR = ".tb2j_resume_cache"


def _file_fingerprint(path):
    st = os.stat(path)
    return [os.path.basename(path), st.st_size, int(st.st_mtime_ns)]


def _signature_payload(exchange):
    """Everything that affects per-energy-point results: the physical/
    numerical parameters (elements, kmesh, Rcut, energy window, ...) plus
    -- if available -- the SIESTA input files actually read, fingerprinted
    by name/size/mtime (not content, to stay cheap). A changed SIESTA
    output will almost always change at least one of these.
    """
    params = dict(
        magnetic_elements=sorted(exchange.magnetic_elements or []),
        index_magnetic_atoms=exchange.index_magnetic_atoms,
        exclude_orbs=sorted(exchange.exclude_orbs or []),
        kmesh=list(exchange._kmesh),
        emin=exchange.emin,
        emax=exchange.emax,
        nz=exchange.nz,
        Rcut=exchange.Rcut,
        efermi=exchange.efermi,
        smearing=exchange.smearing,
        ne=exchange.ne,
        orth=exchange.orth,
        ibz=exchange.ibz,
        orb_decomposition=exchange.orb_decomposition,
        group_by_zeta=exchange.group_by_zeta,
        is_orthogonal=getattr(exchange, "is_orthogonal", None),
    )

    input_files = []
    fdf_fname = getattr(exchange, "fdf_fname", None)
    if fdf_fname and os.path.isfile(fdf_fname):
        input_files.append(_file_fingerprint(fdf_fname))
        d = os.path.dirname(os.path.abspath(fdf_fname)) or "."
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".tshs", ".nc", ".hsx", ".dm", ".xv")):
                input_files.append(_file_fingerprint(os.path.join(d, fn)))

    return {"params": params, "input_files": input_files}


class ResumeCache:
    """Per-energy-point on-disk cache for one exchange calculation.

    :param output_path: the exchange calculation's output directory
        (e.g. "TB2J_results"); the cache lives under
        ``<output_path>/.tb2j_resume_cache/``
    :param exchange: the Exchange(CL2) instance whose parameters/inputs
        are fingerprinted to detect a stale/incompatible cache
    """

    def __init__(self, output_path, exchange):
        self.dir = os.path.join(output_path, CACHE_SUBDIR)
        os.makedirs(self.dir, exist_ok=True)
        self._sigfile = os.path.join(self.dir, "signature.json")
        self._check_or_reset_signature(_signature_payload(exchange))

    def _check_or_reset_signature(self, payload):
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        if os.path.isfile(self._sigfile):
            with open(self._sigfile) as f:
                old = json.load(f)
            if old.get("digest") == digest:
                return
            print(
                "TB2J resume cache does not match the current calculation's "
                "parameters/input files -- discarding stale cached energy "
                "points and starting this calculation from scratch."
            )
            for fn in os.listdir(self.dir):
                if fn.startswith("point_"):
                    os.remove(os.path.join(self.dir, fn))
        with open(self._sigfile, "w") as f:
            json.dump({"digest": digest, "payload": payload}, f, default=str)

    def _point_path(self, idx):
        return os.path.join(self.dir, f"point_{idx:05d}.pickle")

    def has(self, idx):
        return os.path.isfile(self._point_path(idx))

    def load(self, idx, expected_e, tol=1e-10):
        """Return the cached result for point `idx`, or None (and discard
        the stale file) if its stored energy doesn't match `expected_e` --
        a defense-in-depth check in case the signature missed a relevant
        change.
        """
        path = self._point_path(idx)
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            return None
        if abs(complex(data.get("e", None)) - complex(expected_e)) > tol:
            os.remove(path)
            return None
        return data["result"]

    def save(self, idx, e, result):
        """Save one energy point's result, atomically (so a kill mid-write
        never leaves a corrupt point file that a later run would trust)."""
        path = self._point_path(idx)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"e": e, "result": result}, f)
        os.replace(tmp, path)

    def clear(self):
        """Remove the whole cache (called after a fully successful run)."""
        shutil.rmtree(self.dir, ignore_errors=True)
