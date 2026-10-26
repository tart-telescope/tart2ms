# -*- coding: utf-8 -*-
# noqa: E501
# Copyright NRF South African Radio Astronomy Observatory
# Part of Xova BDA averager
# Licensed under GPLv2

import logging
import sys

import numpy as np
import erfa
from astropy.time import Time
from casacore.tables import table as tbl
from progress.bar import FillingSquaresBar as bar
from scipy.constants import c as SPEED_OF_LIGHT

logger = logging.getLogger("tart2ms")


class progress:
    def __init__(self, *args, **kwargs):
        """Wraps a progress bar to check for TTY attachment
        otherwise does prints basic progress periodically
        """
        if sys.stdout.isatty():
            self.__progress = bar(*args, **kwargs)
        else:
            self.__progress = None
            self.__value = 0
            self.__title = args[0]
            self.__max = kwargs.get("max", 1)

    def next(self):
        if self.__progress is None:
            if self.__value % max(int(self.__max * 0.1), 1) == 0:
                logger.info(
                    f"\t {self.__title} progress: "
                    f"{self.__value * 100.0 / self.__max:.0f}%"
                )
            self.__value += 1
        else:
            self.__progress.next()


def baseline_index(a1, a2, no_antennae):
    """
    Computes unique index of a baseline given antenna 1 and antenna 2
    (zero indexed) as input. The arrays may or may not contain
    auto-correlations.

    There is a quadratic series expression relating a1 and a2
    to a unique baseline index(can be found by the double difference
    method)

    Let slow_varying_index be S = min(a1, a2). The goal is to find
    the number of fast varying terms. As the slow
    varying terms increase these get fewer and fewer, because
    we only consider unique baselines and not the conjugate
    baselines)
    B = (-S ^ 2 + 2 * S *  # Ant + S) / 2 + diff between the
    slowest and fastest varying antenna

    :param a1: array of ANTENNA_1 ids
    :param a2: array of ANTENNA_2 ids
    :param no_antennae: number of antennae in the array
    :return: array of baseline ids

    Note: na must be strictly greater than max of 0-indexed
          ANTENNA_1 and ANTENNA_2
    """
    if a1.shape != a2.shape:
        raise ValueError("a1 and a2 must have the same shape!")

    slow_index = np.min(np.array([a1, a2]), axis=0)

    return (slow_index * (-slow_index + (2 * no_antennae + 1))) // 2 + np.abs(a1 - a2)


def dense2sparse_uvw(a1, a2, time, ddid, padded_uvw, ack=True):
    """
    Copy a dense uvw matrix onto a sparse uvw matrix
        a1: sparse antenna 1 index
        a2: sparse antenna 2 index
        time: sparse time
        ddid: sparse data discriptor index
        padded_uvw: a dense ddid-less uvw matrix
                    returned by synthesize_uvw of shape
                    (ntime * nbl, 3), fastest varying
                    by baseline, including auto correlations
    """
    assert time.size == a1.size
    assert a1.size == a2.size
    ants = np.concatenate((a1, a2))
    unique_ants = np.arange(np.max(ants) + 1)
    na = unique_ants.size
    nbl = na * (na - 1) // 2 + na
    unique_time = np.unique(time)
    new_uvw = np.zeros((a1.size, 3), dtype=padded_uvw.dtype)
    outbl = baseline_index(a1, a2, na)

    # Vectorized lookup: map each time value to its index in unique_time.
    # unique_time comes from np.unique (sorted), so searchsorted returns the
    # exact index for every t present in it. This avoids the Python dict and
    # list-comprehension loop while producing identical indices.
    time_indices = np.searchsorted(unique_time, time)
    flat_idx = time_indices * nbl + outbl
    new_uvw[:] = padded_uvw[flat_idx, :]

    return new_uvw


def _check_units(stopctr_units, stopctr_epoch, time_TZ, time_unit, posframe, posunits):
    if [str(x).lower() for x in stopctr_units] != ["rad", "rad"]:
        raise ValueError(f"Unsupported phase centre units {stopctr_units}")
    if str(stopctr_epoch).upper() not in ("J2000", "ICRS"):
        raise ValueError(f"Unsupported phase centre frame {stopctr_epoch}")
    if str(time_TZ).upper() != "UTC" or str(time_unit).lower() != "s":
        raise ValueError(f"Unsupported time reference {time_TZ} [{time_unit}]")
    if str(posframe).upper() != "ITRF" or [str(x).lower() for x in posunits] != ["m"] * 3:
        raise ValueError(f"Unsupported station position frame {posframe} {posunits}")


def itrf_to_celestial(time):
    """
    Rotation matrices taking ITRF vectors to celestial (GCRS ~ J2000/ICRS)
    axes for each MS epoch in ``time`` (UTC MJD seconds).
    Returns an array of shape (time.size, 3, 3).
    """
    t = Time(np.atleast_1d(time) / 86400.0, format="mjd", scale="utc")
    tt, ut1 = t.tt, t.ut1
    # erfa.c2t06a gives GCRS -> ITRS; polar motion (<1 arcsec) is ignored
    c2t = erfa.c2t06a(tt.jd1, tt.jd2, ut1.jd1, ut1.jd2, 0.0, 0.0)
    return np.swapaxes(c2t, -1, -2)


def uvw_basis(phase_dirs):
    """
    Rows are the u, v, w unit vectors (celestial axes) for each
    (ra, dec) phase direction in radians. Returns shape (ndir, 3, 3).
    """
    phase_dirs = np.asarray(phase_dirs, dtype=np.float64).reshape(-1, 2)
    ra, dec = phase_dirs[:, 0], phase_dirs[:, 1]
    sa, ca = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ra)
    return np.stack(
        [
            np.stack([-sa, ca, zero], axis=-1),
            np.stack([-sd * ca, -sd * sa, cd], axis=-1),
            np.stack([cd * ca, cd * sa, sd], axis=-1),
        ],
        axis=1,
    )


def baseline_uvw(station_ECEF, time, a1, a2, phase_dirs, dir_index=None):
    """
    Vectorized UVW for arbitrary (time, a1, a2) rows.

    station_ECEF: ITRF station coordinates (na, 3) in metres
    time: per-row UTC epoch in MJD seconds
    a1, a2: per-row antenna indices
    phase_dirs: (ndir, 2) phase centres (ra, dec) in radians
    dir_index: per-row index into phase_dirs (defaults to 0)

    Uses the same convention as synthesize_uvw: the baseline is
    station[min(a1, a2)] - station[max(a1, a2)].
    """
    station_ECEF = np.asarray(station_ECEF, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    if dir_index is None:
        dir_index = np.zeros(time.size, dtype=int)
    unique_time, time_index = np.unique(time, return_inverse=True)
    rot = itrf_to_celestial(unique_time)  # (ntime, 3, 3)
    lo = np.minimum(a1, a2)
    hi = np.maximum(a1, a2)
    bl_itrf = station_ECEF[lo] - station_ECEF[hi]  # (nrow, 3)
    bl_cel = np.einsum("rij,rj->ri", rot[time_index], bl_itrf)
    basis = uvw_basis(phase_dirs)
    return np.einsum("rij,rj->ri", basis[np.asarray(dir_index)], bl_cel)


def synthesize_uvw(
    station_ECEF,
    time,
    a1,
    a2,
    phase_ref,
    stopctr_units=["rad", "rad"],
    stopctr_epoch="j2000",
    time_TZ="UTC",
    time_unit="s",
    posframe="ITRF",
    posunits=["m", "m", "m"],
    ack=True,
):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis)

    station_ECEF: ITRF station coordinates read from MS::ANTENNA
    time: time column, preferably time centroid
    a1: ANTENNA_1 index
    a2: ANTENNA_2 index
    phase_ref: phase reference centre in radians

    returns dictionary of dense uvw coordinates and indices:
        {
         "UVW": shape (nbl * ntime, 3),
         "TIME_CENTROID": shape (nbl * ntime,),
         "ANTENNA_1": shape (nbl * ntime,),
         "ANTENNA_2": shape (nbl * ntime,)
        }
    Note: input and output antenna indexes may not have the same
          order or be flipped in 1 to 2 index
    """
    assert time.size == a1.size
    assert a1.size == a2.size
    _check_units(stopctr_units, stopctr_epoch, time_TZ, time_unit, posframe, posunits)

    ants = np.concatenate((a1, a2))
    na = np.max(ants) + 1
    nbl = na * (na - 1) // 2 + na
    unique_time = np.unique(time)
    ntime = unique_time.size

    # keep a full uvw array for all antennae - including those
    # dropped by previous calibration and CASA splitting
    antindices = np.stack(np.triu_indices(na, 0), axis=1)
    padded_time = unique_time.repeat(nbl)
    padded_a1 = np.tile(antindices[:, 0], (1, ntime)).ravel()
    padded_a2 = np.tile(antindices[:, 1], (1, ntime)).ravel()
    padded_uvw = baseline_uvw(
        station_ECEF, padded_time, padded_a1, padded_a2, np.asarray(phase_ref)[:1]
    )

    return dict(
        zip(
            ["UVW", "TIME_CENTROID", "ANTENNA1", "ANTENNA2"],
            [padded_uvw, padded_time, padded_a1, padded_a2],
        )
    )


def rephase(vis, uvw, field_ids, sel, freq, pos, refdir, phasesign=-1):
    """
    Rephasor operator
    -- rephases a field to a new phase centre
    freq - in Hz
    pos - (RA, Dec) degree coordinates of the new phase centre for the epoch under
          consideration, either a single pair (shape 2) or one pair per field
          (shape nfield x 2, indexed by field id)
    refdir - array of tuples with original field phase centres in RA and dec at the same epoch as pos (degrees), one per field
    phasesign - should be -1 for the NRAO baseline conventions
    sel - selects a portion of data in uvw, field_ids and vis
    """
    vis_rephase = np.zeros_like(vis)
    pos = np.asarray(pos, dtype=np.float64)
    refdir = np.asarray(refdir, dtype=np.float64)
    if refdir.ndim != 2 or refdir.shape[1] != 2:
        raise ValueError("ref must be shape nfield x 2")
    if pos.shape != (2,) and (pos.ndim != 2 or pos.shape[1] != 2):
        raise ValueError("pos must be shape 2 or nfield x 2")
    if uvw.shape[0] != vis.shape[0]:
        raise ValueError("UVW rows must be the same as vis rows")
    if field_ids.shape[0] != vis.shape[0]:
        raise ValueError("FIELD_ID rows must be the same as vis rows")
    rows = np.flatnonzero(sel)
    fid = field_ids[rows]
    if fid.size > 0 and refdir.shape[0] <= fid.max():
        raise ValueError("Must have at least as many ref positions as unique fields")

    # Per-row direction cosines of the new centre relative to the row's field centre
    ra, dec = np.deg2rad(pos if pos.ndim == 1 else pos[fid]).T
    ra0, dec0 = np.deg2rad(refdir[fid]).T
    d_ra = ra - ra0
    ll = np.cos(dec) * np.sin(d_ra)
    mm = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(d_ra)
    nn = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(d_ra) - 1.0

    # Path difference in metres per row, then phase per (row, freq)
    uvw_sel = uvw[rows, :]
    delay = uvw_sel[:, 0] * ll + uvw_sel[:, 1] * mm + uvw_sel[:, 2] * nn
    inv_wl = np.asarray(freq, dtype=np.float64) / SPEED_OF_LIGHT  # shape (nfreq,)
    x = np.exp(phasesign * 2.0j * np.pi * delay[:, None] * inv_wl[None, :])
    vis_rephase[rows, :, :] = vis[rows, :, :] * x[:, :, None]

    return vis_rephase


def fixms(msname, ack=True):
    """
    Runs an operation similar to the CASA fixvis task
    Recomputes UVW coordinates for the predicted
    az-elev delay projections given a dataset with antenna ICRS
    positions and a time centroid column.
    """
    with tbl(msname + "::ANTENNA", ack=False) as t:
        apos = t.getcol("POSITION")
        aposcoldesc = t.getcoldesc("POSITION")
        posunits = aposcoldesc["keywords"]["QuantumUnits"]
        posframe = aposcoldesc["keywords"]["MEASINFO"]["Ref"]

    with tbl(msname + "::FIELD", ack=False) as t:
        if not np.all(t.getcol("NUM_POLY") == 0):
            logger.critical(
                "UVW recompute does not support"
                " time-variable reference centres."
                " Your dataset will contain averaged"
                " UVW coordinates!"
            )
            return
        fnames = t.getcol("NAME")
        field_stop_ctrs = t.getcol("PHASE_DIR")
        fieldcoldesc = t.getcoldesc("PHASE_DIR")
        stopctr_units = fieldcoldesc["keywords"]["QuantumUnits"]
        stopctr_epoch = fieldcoldesc["keywords"]["MEASINFO"]["Ref"]

    with tbl(msname, ack=False) as t:
        a1 = t.getcol("ANTENNA1")
        a2 = t.getcol("ANTENNA2")
        field_id = t.getcol("FIELD_ID")
        time = t.getcol("TIME_CENTROID")
        timecoldesc = t.getcoldesc("TIME_CENTROID")
        time_TZ = timecoldesc["keywords"]["MEASINFO"]["Ref"]
        time_unit = timecoldesc["keywords"]["QuantumUnits"][0]

    _check_units(stopctr_units, stopctr_epoch, time_TZ, time_unit, posframe, posunits)
    logger.info(f"Computing UVW coordinates for {len(fnames)} field(s)")
    new_uvw = baseline_uvw(
        apos, time, a1, a2, field_stop_ctrs[:, 0, :], dir_index=field_id
    )

    logger.info("Writing computed UVW coordinates to output dataset")

    with tbl(msname, ack=False, readonly=False) as t:
        t.lock()  # workaround dask-ms bug not releasing user locks
        t.putcol("UVW", new_uvw)
        t.unlock()
