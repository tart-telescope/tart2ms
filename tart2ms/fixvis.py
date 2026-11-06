# -*- coding: utf-8 -*-
# noqa: E501
# Copyright NRF South African Radio Astronomy Observatory
# Part of Xova BDA averager
# Licensed under GPLv2

import logging
import sys

import numpy as np
from progress.bar import FillingSquaresBar as bar

import erfa
from astropy import units as u
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
from casacure.quanta import constants, quantity
from casacure.tables import table as tbl

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


def _coldesc_units(v):
    """QuantumUnits from a column descriptor, as a plain list of unit strings.

    python-casacore returns a plain list (``['m', 'm', 'm']``); casacure
    returns array-valued keywords written through daskms as records
    (``{'shape': array([3]), 'array': ['m', 'm', 'm']}``). Normalise both
    (and a bare scalar unit) to a list.
    """
    if isinstance(v, dict):
        v = v.get("array", [])
    if isinstance(v, str):
        return [v]
    return list(v)


def _min_rotation(a, b):
    """The minimal rotation (Rodrigues) taking unit vector a onto unit vector b."""
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-15:
        return np.eye(3)
    kx = np.array(
        [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]]
    ) / s
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)


def _itrf_uvw_frame(mjd, ra, dec):
    """Unit (u, v, w) axes in ITRF for the J2000 direction (ra, dec) at epoch.

    Replicates what casacore's ``measures.to_uvw`` used: the baseline is
    transported ITRF -> J2000 by the frame chain (CIO precession-nutation,
    Earth rotation, polar motion -- erfa), and the whole frame is then
    rotated so the source lands on its *apparent* place (annual aberration
    and light deflection, via astropy's ICRS -> ITRS direction transform).

    Validated against python-casacore 3.8.1's ``to_uvw()``: the frame agrees
    to better than ~1.7 arcsec of rotation, i.e. under 2e-5 m on TART's
    baselines (python-casacore's exact internal chain is not reproducible
    from public APIs; this is far below TART's imaging thresholds).

    mjd: epoch in days (MJD, scale from the caller's TIME column).
    ra, dec: phase reference in radians (J2000).
    """
    d1, d2 = 2400000.5, float(mjd)
    c2i = erfa.c2i06a(d1, d2)                            # ICRS -> true of date
    pom = erfa.pom00(0.0, 0.0, erfa.sp00(d1, d2))        # polar motion (xp=yp=0)
    g = pom @ erfa.rz(erfa.era00(d1, d2), np.eye(3)) @ c2i   # ICRS -> ITRF

    shat = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array(
        [
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        ]
    )

    s_geo = g @ shat
    c = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
    it = c.transform_to(ITRS(obstime=Time(mjd, format="mjd", scale="utc")))
    s_app = np.array(
        [
            it.cartesian.x.to_value(),
            it.cartesian.y.to_value(),
            it.cartesian.z.to_value(),
        ]
    )
    s_app /= np.linalg.norm(s_app)

    f = _min_rotation(s_geo, s_app)
    return f @ (g @ east), f @ (g @ north), s_app


_TIME_SCALES = {"UTC": "utc", "TAI": "tai", "IAT": "tai", "GPS": "gps", "TT": "tt"}


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
    Note: The UVW frame is rebuilt per timestamp with erfa/astropy
          (see _itrf_uvw_frame); there is no casacore.measures state,
          so unlike the old pyrap implementation this is not subject
          to casacore's measures thread-safety restriction.
    """
    assert time.size == a1.size
    assert a1.size == a2.size
    if posframe.upper() not in ("ITRF", "ITRS"):
        raise NotImplementedError(
            f"UVW synthesis assumes ITRF antenna positions, got {posframe!r}"
        )
    if stopctr_epoch.lower() not in ("j2000", "icrs"):
        raise NotImplementedError(
            f"phase reference must be J2000/ICRS, got {stopctr_epoch!r}"
        )

    ants = np.concatenate((a1, a2))
    unique_ants = np.arange(np.max(ants) + 1)
    unique_time = np.unique(time)
    na = unique_ants.size
    nbl = na * (na - 1) // 2 + na
    ntime = unique_time.size

    # keep a full uvw array for all antennae - including those
    # dropped by previous calibration and CASA splitting
    padded_uvw = np.zeros((ntime * nbl, 3), dtype=np.float64)
    antindices = np.stack(np.triu_indices(na, 0), axis=1)
    padded_time = unique_time.repeat(nbl)
    padded_a1 = np.tile(antindices[:, 0], (1, ntime)).ravel()
    padded_a2 = np.tile(antindices[:, 1], (1, ntime)).ravel()

    # phase reference -> J2000 radians
    ra = u.Quantity(phase_ref[0, 0], stopctr_units[0]).to_value(u.rad)
    dec = u.Quantity(phase_ref[0, 1], stopctr_units[1]).to_value(u.rad)
    scale = _TIME_SCALES.get(str(time_TZ).upper(), "utc")

    if ack:
        p = progress("Calculating UVW", max=unique_time.size)
    for ti, t in enumerate(unique_time):
        if ack:
            p.next()
        # TIME column values are seconds (etc) since MJD 0 in the column's
        # MEASINFO reference scale; convert to MJD days for the frame.
        mjd = u.Quantity(t, time_unit).to_value(u.day)
        if scale != "utc":
            mjd = Time(mjd, format="mjd", scale=scale).utc.mjd
        u_vec, v_vec, w_vec = _itrf_uvw_frame(mjd, ra, dec)

        # UVW of every antenna position, then difference per baseline.
        # Same convention as in CASA (Convention for UVW calculations in
        # CASA, Rau 2013). Broadcasting over all baselines avoids the
        # per-baseline Python loop and produces identical results.
        station_uv = np.column_stack(
            [
                station_ECEF @ u_vec,
                station_ECEF @ v_vec,
                station_ECEF @ w_vec,
            ]
        )
        bl_a1 = antindices[:, 0]
        bl_a2 = antindices[:, 1]
        padded_uvw[ti * nbl : (ti + 1) * nbl, :] = station_uv[bl_a1] - station_uv[bl_a2]

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
    pos - tupple containing degree coordinates for RA and Dec for the epoch under consideration
    refdir - array of tuples with original field phase centres in RA and dec at the same epoch as pos (degrees), one per field
    phasesign - should be -1 for the NRAO baseline conventions
    sel - selects a portion of data in uvw, field_ids and vis
    """
    vis_rephase = np.zeros_like(vis)
    uniq_fields = np.unique(field_ids[sel])
    if refdir.shape[0] < uniq_fields.max() if uniq_fields.size > 0 else 0:
        raise ValueError("Must have at least as many ref positions as unique fields")
    if refdir.shape[1] != 2:
        raise ValueError("ref must be shape nfield x 2")
    if pos.size != 2 and pos.shape[0] != 2:
        raise ValueError("pos must be shape 2")
    if uvw.shape[0] != vis.shape[0]:
        raise ValueError("UVW rows must be the same as vis rows")
    if field_ids.shape[0] != vis.shape[0]:
        raise ValueError("FIELD_ID rows must be the same as vis rows")
    for ifid, fid in enumerate(uniq_fields):
        selfid = np.logical_and(field_ids == fid, sel)
        nrowsel = np.sum(selfid)
        cos = np.cos
        sin = np.sin
        sqrt = np.sqrt
        ra, dec = np.deg2rad(pos)
        ra0, dec0 = np.deg2rad(refdir[fid])
        d_ra = ra - ra0
        d_dec = dec
        d_decp = dec0
        c_d_dec = cos(d_dec)
        s_d_dec = sin(d_dec)
        s_d_ra = sin(d_ra)
        c_d_ra = cos(d_ra)
        c_d_decp = cos(d_decp)
        s_d_decp = sin(d_decp)
        ll = c_d_dec * s_d_ra
        mm = s_d_dec * c_d_decp - c_d_dec * s_d_decp * c_d_ra
        nn = s_d_dec * s_d_decp + c_d_dec * c_d_decp * c_d_ra - 1.0

        wl = constants["c"].get_value() / freq  # shape (nfreq,)
        # Use broadcasting instead of creating full nrowsel x nfreq x 3 copies
        # uvw has shape (nrowsel, 3), wl has shape (nfreq,)
        uvw_sel = uvw[selfid, :]  # (nrowsel, 3)
        uu = uvw_sel[:, 0:1] / wl[None, :]  # (nrowsel, nfreq) via broadcasting
        vv = uvw_sel[:, 1:2] / wl[None, :]
        ww = uvw_sel[:, 2:3] / wl[None, :]

        x = np.exp(phasesign * 2.0j * np.pi * (uu * ll + vv * mm + ww * nn))
        ncorr = vis.shape[2]
        # x has shape (nrowsel, nfreq) -- broadcast to (nrowsel, nfreq, ncorr)
        vis_rephase[selfid, :, :] = vis[selfid, :, :] * x[:, :, None]

    return vis_rephase


def fixms(msname, ack=True):
    """
    Runs an operation similar to the CASA fixvis task
    Recomputes UVW coordinates for the predicted
    az-elev delay projections given a dataset with antenna ITRF
    positions and a time centroid column.

    Note: the UVW frame is rebuilt per timestamp with erfa/astropy
    (see _itrf_uvw_frame); there is no shared casacore.measures state,
    so unlike the old pyrap implementation this is not subject to
    casacore's measures thread-safety restriction.
    """
    with tbl(msname + "::ANTENNA", ack=False) as t:
        apos = t.getcol("POSITION")
        aposcoldesc = t.getcoldesc("POSITION")
        posunits = _coldesc_units(aposcoldesc["keywords"]["QuantumUnits"])
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
        stopctr_units = _coldesc_units(fieldcoldesc["keywords"]["QuantumUnits"])
        stopctr_epoch = fieldcoldesc["keywords"]["MEASINFO"]["Ref"]

    with tbl(msname, ack=False) as t:
        a1 = t.getcol("ANTENNA1")
        a2 = t.getcol("ANTENNA2")
        uvw = t.getcol("UVW")
        field_id = t.getcol("FIELD_ID")
        ddid = t.getcol("DATA_DESC_ID")
        time = t.getcol("TIME_CENTROID")
        timecoldesc = t.getcoldesc("TIME_CENTROID")
        time_TZ = timecoldesc["keywords"]["MEASINFO"]["Ref"]
        time_unit = _coldesc_units(timecoldesc["keywords"]["QuantumUnits"])[0]

    logger.info("Computing UVW coordinates for output dataset... WAIT")
    new_uvw = np.zeros_like(uvw, dtype=uvw.dtype)
    n_fields = len(fnames)
    for fi in range(n_fields):
        fsel = field_id == fi
        padded_uvw = synthesize_uvw(
            station_ECEF=apos,
            time=time[fsel],
            a1=a1[fsel],
            a2=a2[fsel],
            phase_ref=field_stop_ctrs[fi],
            stopctr_units=stopctr_units,
            time_TZ=time_TZ,
            time_unit=time_unit,
            stopctr_epoch=stopctr_epoch,
            posframe=posframe,
            posunits=posunits,
            ack=ack,
        )
        new_uvw[fsel] = dense2sparse_uvw(
            a1=a1[fsel],
            a2=a2[fsel],
            time=time[fsel],
            ddid=ddid[fsel],
            padded_uvw=padded_uvw["UVW"],
            ack=ack,
        )
        logger.info(
            f"\t {fi + 1} / {n_fields} field {fnames[fi]} {field_stop_ctrs[fi]} completed"
        )

    logger.info("Writing computed UVW coordinates to output dataset")

    with tbl(msname, ack=False, readonly=False) as t:
        t.lock()  # workaround dask-ms bug not releasing user locks
        t.putcol("UVW", new_uvw)
        t.unlock()
