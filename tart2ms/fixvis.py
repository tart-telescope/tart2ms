# -*- coding: utf-8 -*-
# noqa: E501
# Copyright NRF South African Radio Astronomy Observatory
# Part of Xova BDA averager
# Licensed under GPLv2

from pyrap.tables import table as tbl
import numpy as np
from pyrap.measures import measures
from pyrap.quanta import quantity

import logging
from progress.bar import FillingSquaresBar as bar
import sys

logger = logging.getLogger()

class progress():
    def __init__(self, *args, **kwargs):
        """ Wraps a progress bar to check for TTY attachment
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
            if self.__value % int(self.__max * 0.1) == 0:
                logger.info(f"\t {self.__title} progress: "
                            f"{self.__value *100. / self.__max:.0f}%")
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

    return (slow_index * (-slow_index + (2 * no_antennae + 1))) // 2 + \
        np.abs(a1 - a2)


def dense2sparse_uvw(a1, a2, time, ddid, padded_uvw):
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
    p = progress('Copying UVW to dataset', max=a1.size)
    for outrow in range(a1.size):
        p.next()
        lookupt = np.argwhere(unique_time == time[outrow])
        # note: uvw same for all ddid (in m)
        new_uvw[outrow][:] = padded_uvw[lookupt * nbl + outbl[outrow], :]

    return new_uvw


def synthesize_uvw(station_ECEF, time, a1, a2,
                   phase_ref,
                   stopctr_units=["rad", "rad"], stopctr_epoch="j2000",
                   time_TZ="UTC", time_unit="s",
                   posframe="ITRF", posunits=["m", "m", "m"]):
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
    Note: This operation CANNOT be applied blockwise due
          to a casacore.measures threadsafety issue
    """
    assert time.size == a1.size
    assert a1.size == a2.size

    ants = np.concatenate((a1, a2))
    unique_ants = np.arange(np.max(ants) + 1)
    unique_time = np.unique(time)
    na = unique_ants.size
    nbl = na * (na - 1) // 2 + na
    ntime = unique_time.size

    # keep a full uvw array for all antennae - including those
    # dropped by previous calibration and CASA splitting
    padded_uvw = np.zeros((ntime * nbl, 3), dtype=np.float64)
    antindices = np.stack(np.triu_indices(na, 0),
                          axis=1)
    padded_time = unique_time.repeat(nbl)
    padded_a1 = np.tile(antindices[:, 0], (1, ntime)).ravel()
    padded_a2 = np.tile(antindices[:, 1], (1, ntime)).ravel()

    dm = measures()
    epoch = dm.epoch(time_TZ, quantity(time[0], time_unit))
    refdir = dm.direction(stopctr_epoch,
                          quantity(phase_ref[0, 0], stopctr_units[0]),
                          quantity(phase_ref[0, 1], stopctr_units[1]))
    obs = dm.position(posframe,
                      quantity(station_ECEF[0, 0], posunits[0]),
                      quantity(station_ECEF[0, 1], posunits[1]),
                      quantity(station_ECEF[0, 2], posunits[2]))

    # setup local horizon coordinate frame with antenna 0 as reference position
    dm.do_frame(obs)
    dm.do_frame(refdir)
    dm.do_frame(epoch)
    p = progress('Calculating UVW', max=unique_time.size)
    for ti, t in enumerate(unique_time):
        p.next()
        epoch = dm.epoch("UT1", quantity(t, "s"))
        dm.do_frame(epoch)

        station_uv = np.zeros_like(station_ECEF)
        for iapos, apos in enumerate(station_ECEF):
            compuvw = dm.to_uvw(dm.baseline(posframe,
                                            quantity([apos[0],
                                                      station_ECEF[0, 0]],
                                                     posunits[0]),
                                            quantity([apos[1],
                                                      station_ECEF[0, 1]],
                                                     posunits[1]),
                                            quantity([apos[2],
                                                      station_ECEF[0, 2]],
                                                     posunits[2])))
            station_uv[iapos] = compuvw["xyz"].get_value()[0:3]
        for bl in range(nbl):
            blants = antindices[bl]
            bla1 = blants[0]
            bla2 = blants[1]
            # same as in CASA convention (Convention for UVW calculations
            # in CASA, Rau 2013)
            padded_uvw[ti*nbl + bl, :] = station_uv[bla1] - station_uv[bla2]

    return dict(zip(["UVW", "TIME_CENTROID", "ANTENNA1", "ANTENNA2"],
                    [padded_uvw, padded_time, padded_a1, padded_a2]))


def fixms(msname):
    """
        Runs an operation similar to the CASA fixvis task
        Recomputes UVW coordinates with casacore for the predicted
        az-elev delay projections given a dataset with antenna ICRS
        positions and a time centroid column.

        Note: This operation CANNOT be applied blockwise due
        to a casacore.measures threadsafety issue
    """
    with tbl(msname + "::ANTENNA", ack=False) as t:
        apos = t.getcol("POSITION")
        aposcoldesc = t.getcoldesc("POSITION")
        posunits = aposcoldesc["keywords"]["QuantumUnits"]
        posframe = aposcoldesc["keywords"]["MEASINFO"]["Ref"]

    with tbl(msname + "::FIELD", ack=False) as t:
        if not np.all(t.getcol("NUM_POLY") == 0):
            logger.critical("UVW recompute does not support"
                            " time-variable reference centres."
                            " Your dataset will contain averaged"
                            " UVW coordinates!")
            return
        fnames = t.getcol("NAME")
        field_stop_ctrs = t.getcol("PHASE_DIR")
        fieldcoldesc = t.getcoldesc("PHASE_DIR")
        stopctr_units = \
            fieldcoldesc["keywords"]["QuantumUnits"]
        stopctr_epoch = \
            fieldcoldesc["keywords"]["MEASINFO"]["Ref"]

    with tbl(msname, ack=False) as t:
        a1 = t.getcol("ANTENNA1")
        a2 = t.getcol("ANTENNA2")
        uvw = t.getcol("UVW")
        field_id = t.getcol("FIELD_ID")
        ddid = t.getcol("DATA_DESC_ID")
        time = t.getcol("TIME_CENTROID")
        timecoldesc = t.getcoldesc("TIME_CENTROID")
        time_TZ = timecoldesc["keywords"]["MEASINFO"]["Ref"]
        time_unit = timecoldesc["keywords"]["QuantumUnits"][0]

    logger.info("Computing UVW coordinates for output dataset... WAIT")
    new_uvw = np.zeros_like(uvw, dtype=uvw.dtype)
    for fi in range(len(fnames)):
        fsel = field_id == fi
        padded_uvw = synthesize_uvw(station_ECEF=apos,
                                    time=time[fsel],
                                    a1=a1[fsel],
                                    a2=a2[fsel],
                                    phase_ref=field_stop_ctrs[fi],
                                    stopctr_units=stopctr_units,
                                    time_TZ=time_TZ,
                                    time_unit=time_unit,
                                    stopctr_epoch=stopctr_epoch,
                                    posframe=posframe,
                                    posunits=posunits)
        new_uvw[fsel] = dense2sparse_uvw(a1=a1[fsel],
                                         a2=a2[fsel],
                                         time=time[fsel],
                                         ddid=ddid[fsel],
                                         padded_uvw=padded_uvw["UVW"])
        logger.info("\t {} / {} fields completed".format(fi + 1, len(fnames)))

    logger.info("Writing computed UVW coordinates to output dataset")

    with tbl(msname, ack=False, readonly=False) as t:
        t.lock()  # workaround dask-ms bug not releasing user locks
        t.putcol("UVW", new_uvw)
        t.unlock()
