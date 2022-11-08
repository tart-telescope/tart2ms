'''
    Get TART data into a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019-2022.

    Official Documentation is Here.
        https://casa.nrao.edu/Memos/229.html#SECTION00044000000000000000

    License. GPLv3.
'''
import logging
import json
import h5py
import dask
import dateutil

import dask.array as da
import numpy as np


from itertools import product

from casacore.quanta import quantity

from astropy import coordinates as ac

from daskms import Dataset, xds_to_table

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle
from astropy.constants import R_earth


from tart.operation import settings
from tart.imaging.visibility import Visibility
from tart.imaging import calibration

from tart_tools import api_imaging
from .fixvis import (fixms,
                     synthesize_uvw,
                     dense2sparse_uvw,
                     progress,
                     rephase)

from .util import rayleigh_criterion

LOGGER = logging.getLogger()

'''
The following from Oleg Smirnov.

>
> * We have a single circular polarization that I've labelled 1
> (from FITS scheme I googled somewhere. Could use RR I guess).

Ah, my favourite bit of the MS documentation! According to
https://casa.nrao.edu/Memos/229.html#SECTION000613000000000000000, we
have CORR_TYPE: An integer for each correlation product indicating the
Stokes type as defined in the Stokes class enumeration.

And where does a naive user find this mysterious "Stokes class
enumeration"? Why, it's in the C++ source code itself! Voila:
https://casa.nrao.edu/active/docs/doxygen/html/classcasa_1_1Stokes.html.
And they don't even give you the numbers! You have to use your fingers
to count them off, old school. And how were you supposed to know about
this? I have no idea -- the only reason I know about this is because I
worked with the AIPS++ codebase back in the day...

Anyway, here's a Python list version of the enumeration, if it helps:
'''
MS_STOKES_ENUMS = {
    "Undefined": 0,
    "I": 1,
    "Q": 2,
    "U": 3,
    "V": 4,
    "RR": 5,
    "RL": 6,
    "LR": 7,
    "LL": 8,
    "XX": 9,
    "XY": 10,
    "YX": 11,
    "YY": 12,
    "RX": 13,
    "RY": 14,
    "LX": 15,
    "LY": 16,
    "XR": 17,
    "XL": 18,
    "YR": 19,
    "YL": 20,
    "PP": 21,
    "PQ": 22,
    "QP": 23,
    "QQ": 24,
    "RCircular": 25,
    "LCircular": 26,
    "Linear": 27,
    "Ptotal": 28,
    "Plinear": 29,
    "PFtotal": 30,
    "PFlinear": 31,
    "Pangle": 32}

# These are Jones vectors in a linearly polarized basis [E_x, E_y]
_ZZ = 1.0 / np.sqrt(2)
POL_RESPONSES = {
    'XX': [1.0, 0.0],
    'YY': [0.0, 1.0],
    'RR': [_ZZ, -_ZZ*1.0j],
    'LL': [_ZZ, +_ZZ*1.0j]
}


class MSTable:
    '''
        Little Helper to simplify writing a table.
    '''

    def __init__(self, ms_name, table_name):
        self.table_name = "::".join((ms_name, table_name))
        self.datasets = []

    def append(self, dataset):
        self.datasets.append(dataset)

    def write(self):
        writes = xds_to_table(self.datasets, self.table_name, columns="ALL")
        dask.compute(writes)


def timestamp_to_ms_epoch(t_stamp):
    ''' Convert an timestamp to seconds (epoch values)
        epoch suitable for using in a Measurement Set

    Parameters
    ----------
    t_stamp : A timestamp object.

    Returns
    -------
    t : float
      The epoch time ``t`` in seconds suitable for fields in
      measurement sets.
    '''
    return quantity(t_stamp.isoformat()).get_value("s")


def ms_create(ms_table_name, info,
              ant_pos, vis_array,
              baselines,
              timestamps,
              pol_feeds,
              sources,
              phase_center_policy,
              override_telescope_name,
              uvw_generator='casacore'):
    ''' Create a Measurement Set from some TART observations

    Parameters
    ----------

    ms_table_name : string
        The name of the MS top level directory. I think this only workds in
        the local directory.

    info : JSON
        "info": {
            "info": {
                "L0_frequency": 1571328000.0,
                "bandwidth": 2500000.0,
                "baseband_frequency": 4092000.0,
                "location": {
                    "alt": 270.0,
                    "lat": -45.85177,
                    "lon": 170.5456
                },
                "name": "Signal Hill - Dunedin",
                "num_antenna": 24,
                "operating_frequency": 1575420000.0,
                "sampling_frequency": 16368000.0
            }
        },

    Returns
    -------
    None

    '''
    try:
        loc = info['location']
    except Exception:
        loc = info

    lat, lon, height = loc["lat"], loc["lon"], loc["alt"]
    LOGGER.info("Telescope position (WGS84):")
    LOGGER.info(f"\tLat {lat}")
    LOGGER.info(f"\tLon {lon}")
    LOGGER.info(f"\tAlt {height}")

    epoch_s = list(map(timestamp_to_ms_epoch, timestamps))
    LOGGER.debug(f"Time {epoch_s}")
    LOGGER.info(f"Min time: {np.min(timestamps)} -- {np.min(epoch_s)}")
    LOGGER.info(f"Max time: {np.max(timestamps)} -- {np.max(epoch_s)}")

    # Sort out the coordinate frames using astropy
    # https://casa.nrao.edu/casadocs/casa-5.4.1/reference-material/coordinate-frames
    # iers.conf.iers_auto_url = 'https://astroconda.org/aux/astropy_mirror/iers_a_1/finals2000A.all'
    # iers.conf.auto_max_age = None

    location = EarthLocation.from_geodetic(lon=loc['lon']*u.deg,
                                           lat=loc['lat']*u.deg,
                                           height=loc['alt']*u.m,
                                           ellipsoid='WGS84')
    obstime = Time(timestamps)
    LOGGER.debug(f"obstime {obstime}")

    # local_frame = AltAz(obstime=obstime, location=location)
    # LOGGER.info(f"local_frame {local_frame}")

    phase_altaz = SkyCoord(alt=[90.0*u.deg]*len(obstime), az=[0.0*u.deg]*len(obstime),
                           obstime=obstime, frame='altaz', location=location)
    phase_j2000 = phase_altaz.transform_to('fk5')
    LOGGER.debug(f"phase_j2000 {phase_j2000}")

    # Get the stokes enums for the polarization types
    corr_types = [[MS_STOKES_ENUMS[p_f] for p_f in pol_feeds]]

    LOGGER.info("Pol Feeds {}".format(pol_feeds))
    LOGGER.debug("Correlation Types {}".format(corr_types))
    num_freq_channels = [1]

    ant_table = MSTable(ms_table_name, 'ANTENNA')
    feed_table = MSTable(ms_table_name, 'FEED')
    field_table = MSTable(ms_table_name, 'FIELD')
    pol_table = MSTable(ms_table_name, 'POLARIZATION')
    obs_table = MSTable(ms_table_name, 'OBSERVATION')
    # SOURCE is an optional MS sub-table
    src_table = MSTable(ms_table_name, 'SOURCE')

    ddid_table_name = "::".join((ms_table_name, "DATA_DESCRIPTION"))
    spw_table_name = "::".join((ms_table_name, "SPECTRAL_WINDOW"))

    ms_datasets = []
    ddid_datasets = []
    spw_datasets = []

    # Create ANTENNA dataset
    # Each column in the ANTENNA has a fixed shape so we
    # can represent all rows with one dataset
    num_ant = len(ant_pos)

    # Now convert each antenna location to ECEF coordinates for the measurement set
    # This is laborious but seems to work.
    #
    # Zero is due north.
    ant_posang = [Angle(np.arctan2(a[0], a[1]), unit=u.rad) for a in ant_pos]
    ant_s = [np.sqrt(a[0]*a[0] + a[1]*a[1]) for a in ant_pos]
    ant_distance = [(s / R_earth.value) for s in ant_s]

    ant_lon_lat = [ac.offset_by(lon=lon*u.deg, lat=lat*u.deg, posang=theta, distance=d)
                   for theta, d in zip(ant_posang, ant_distance)]
    ant_locations = [EarthLocation.from_geodetic(lon=lon,  lat=lat, height=loc['alt']*u.m,  ellipsoid='WGS84')
                     for lon, lat in ant_lon_lat]
    ant_positions = [[e.x.value, e.y.value, e.z.value] for e in ant_locations]
    antenna_itrf_pos = position = da.asarray(ant_positions)

    # Antenna diameter in meters
    diameter = da.ones(num_ant) * 0.025
    offset = da.zeros((num_ant, 3))
    names = np.array(['ANTENNA-%d' % i for i in range(num_ant)], dtype=object)
    stations = np.array([info['name'] for i in range(num_ant)], dtype=object)
    stationtype = np.array(["GROUND-BASED"] * num_ant, dtype=object)
    stationmount = np.array(["X-Y"] * num_ant, dtype=object)
    dataset = Dataset({
        'POSITION': (("row", "xyz"), position),
        'OFFSET': (("row", "xyz"), offset),
        'DISH_DIAMETER': (("row",), diameter),
        'NAME': (("row",), da.from_array(names, chunks=num_ant)),
        'STATION': (("row",), da.from_array(stations, chunks=num_ant)),
        'TYPE': (("row",), da.from_array(stationtype, chunks=num_ant)),
        'MOUNT': (("row",), da.from_array(stationmount, chunks=num_ant)),
    })
    ant_table.append(dataset)

    # Create a FEED dataset
    # There is one feed per antenna, so this should be quite similar to the ANTENNA
    num_pols = len(pol_feeds)
    pol_types = pol_feeds
    pol_responses = [POL_RESPONSES[ct] for ct in pol_feeds]

    LOGGER.debug("Pol Types {}".format(pol_types))
    LOGGER.debug("Pol Responses {}".format(pol_responses))

    antenna_ids = da.asarray(range(num_ant))
    feed_ids = da.zeros(num_ant)
    num_receptors = da.zeros(num_ant) + num_pols
    polarization_types = np.array(
        [pol_types for i in range(num_ant)], dtype=object)
    receptor_angles = np.array([[0.0] for i in range(num_ant)])
    pol_response = np.array([pol_responses for i in range(num_ant)])

    beam_offset = np.array([[[0.0, 0.0]] for i in range(num_ant)])

    dataset = Dataset({
        'ANTENNA_ID': (("row",), antenna_ids),
        'FEED_ID': (("row",), feed_ids),
        'NUM_RECEPTORS': (("row",), num_receptors),
        'POLARIZATION_TYPE': (("row", "receptors",),
                              da.from_array(polarization_types, chunks=num_ant)),
        'RECEPTOR_ANGLE': (("row", "receptors",),
                           da.from_array(receptor_angles, chunks=num_ant)),
        'POL_RESPONSE': (("row", "receptors", "receptors-2"),
                         da.from_array(pol_response, chunks=num_ant)),
        'BEAM_OFFSET': (("row", "receptors", "radec"),
                        da.from_array(beam_offset, chunks=num_ant)),
    })
    feed_table.append(dataset)

    # ------------------------ FIELD dataset -------------------------- #
    LOGGER.info(f"Setting phase center per {phase_center_policy}")
    direction = np.array([[phase_j2000.ra.radian, phase_j2000.dec.radian]])
    assert direction.ndim == 3
    assert direction.shape[0] == 1
    assert direction.shape[1] == 2
    
    
    def __twelveball(direction):
        """ standardized Jhhmmss-ddmmss name """
        sc_dir = SkyCoord(direction[0]*u.rad, direction[1]*u.rad, frame='icrs')
        sign = "-" if sc_dir.dec.dms[0] < 0 else "+"
        sc_dir_repr = f"J{sc_dir.ra.hms[0]:02.0f}{sc_dir.ra.hms[1]:02.0f}{sc_dir.ra.hms[2]:02.0f}"\
                      f"{sign}"\
                      f"{abs(sc_dir.dec.dms[0]):02.0f}{abs(sc_dir.dec.dms[1]):02.0f}{abs(sc_dir.dec.dms[2]):02.0f}"
        return sc_dir_repr

    directions = direction.T
    for d in directions:
        LOGGER.info(f"    shapshot direction {d[0]}, {d[1]} {SkyCoord(d[0]*u.rad, d[1]*u.rad, frame='icrs').to_string('dms')}")


    if phase_center_policy == "instantaneous-zenith":
        pass
    elif (phase_center_policy == "no-rephase-obs-midpoint") or \
         (phase_center_policy == "rephase-obs-midpoint"):
        direction = direction[:, :, direction.shape[2]//2].reshape(1, 2, 1)  # observation midpoint
    elif phase_center_policy == "rephase-SCP":
        direction = np.array([0, np.deg2rad(-90)]).reshape(1, 2, 1)
    elif phase_center_policy == "rephase-NCP":
        direction = np.array([0, np.deg2rad(90)]).reshape(1, 2, 1)
    else:
        raise ValueError(f"phase_center_policy must be one of "
                         f"['instantaneous-zenith','rephase-obs-midpoint','rephase-SCP','rephase-NCP',"
                         f"'no-rephase-obs-midpoint'] got {phase_center_policy}")
    field_name = da.asarray(np.array(list(map(__twelveball, direction.reshape(2, direction.shape[2]).T)),
                                     dtype=object), chunks=1)
    field_direction = da.asarray(
        direction.T.reshape(direction.shape[2],
                            1, 2).copy(), chunks=(1, None, None))  # nrow x npoly x 2

    # zeroth order polynomial in time for phase center.
    field_num_poly = da.zeros(direction.shape[2], chunks=1)
    dir_dims = ("row", 'field-poly', 'field-dir',)

    dataset = Dataset({
        'PHASE_DIR': (dir_dims, field_direction),
        'DELAY_DIR': (dir_dims, field_direction),
        'REFERENCE_DIR': (dir_dims, field_direction),
        'NUM_POLY': (("row", ), field_num_poly),
        'NAME': (("row", ), field_name),
    })
    field_table.append(dataset)

    # ------------------------ OBSERVATION dataset ---------------------------- #
    LOGGER.info(f"Writing MS for telescope name '{override_telescope_name}'")
    dataset = Dataset({
        'TELESCOPE_NAME': (("row",), da.asarray(np.asarray([override_telescope_name], dtype=object), chunks=1)),
        'OBSERVER': (("row",), da.asarray(np.asarray(['Tim'], dtype=object), chunks=1)),
        "TIME_RANGE": (("row", "obs-exts"), da.asarray(np.array([[epoch_s[0], epoch_s[-1]]]), chunks=1))
    })
    obs_table.append(dataset)

    # ----------------------------- SOURCE datasets -------------------------- #
    if sources:
        if len(epoch_s) != len(sources):
            raise RuntimeError(
                "If sources are specified then we expected epochs to be of same size as sources list")
        for epoch_s_i, sources_i in zip(epoch_s, sources):
            for src in sources_i:
                name = src['name']
                # Convert to J2000
                dir_altaz = SkyCoord(alt=src['el']*u.deg, az=src['az']*u.deg, obstime=obstime[0],
                                     frame='altaz', location=location)
                dir_j2000 = dir_altaz.transform_to('fk5')
                direction_src = [dir_j2000.ra.radian, dir_j2000.dec.radian]
                LOGGER.debug(
                    f"SOURCE: {name}, timestamp: {timestamps}, dir: {direction_src}")
                # , 1, dtype=np.int32)
                dask_num_lines = da.asarray(np.asarray([1], dtype=np.int32))
                dask_direction = da.asarray(np.asarray(
                    direction_src, dtype=np.float64), chunks=1)[None, :]
                dask_name = da.asarray(np.asarray(
                    [name], dtype=object), chunks=1)
                dask_time = da.asarray(np.asarray(
                    [epoch_s_i], dtype=object), chunks=1)
                dataset = Dataset({
                    "NUM_LINES": (("row",), dask_num_lines),
                    "NAME": (("row",), dask_name),
                    "TIME": (("row",), dask_time),
                    "DIRECTION": (("row", "dir"), dask_direction),
                })
                src_table.append(dataset)

    # Create POLARISATION datasets.
    # Dataset per output row required because column shapes are variable

    for corr_type in corr_types:
        corr_prod = [[i, i] for i in range(len(corr_type))]

        corr_prod = np.array(corr_prod)
        LOGGER.debug("Corr Prod {}".format(corr_prod))
        LOGGER.debug("Corr Type {}".format(corr_type))

        dask_num_corr = da.full((1,), len(corr_type), dtype=np.int32)
        LOGGER.debug("NUM_CORR {}".format(dask_num_corr))
        dask_corr_type = da.from_array(corr_type,
                                       chunks=len(corr_type))[None, :]
        dask_corr_product = da.asarray(corr_prod)[None, :]
        LOGGER.debug("Dask Corr Prod {}".format(dask_corr_product.shape))
        LOGGER.debug("Dask Corr Type {}".format(dask_corr_type.shape))
        dataset = Dataset({
            "NUM_CORR": (("row",), dask_num_corr),
            "CORR_TYPE": (("row", "corr"), dask_corr_type),
            "CORR_PRODUCT": (("row", "corr", "corrprod_idx"), dask_corr_product),
        })

        pol_table.append(dataset)

    # Create multiple SPECTRAL_WINDOW datasets
    # Dataset per output row required because column shapes are variable
    spw_chan_freqs = []
    for spw_i, num_chan in enumerate(num_freq_channels):
        dask_num_chan = da.full((1,), num_chan, dtype=np.int32, chunks=(1,))
        spw_chan_freqs.append(np.array([info['operating_frequency']]))
        dask_chan_freq = da.asarray(
            [[info['operating_frequency']]], chunks=(1, None))
        dask_chan_width = da.full(
            (1, num_chan), 2.5e6/num_chan, chunks=(1, None))
        spw_name = da.asarray(
            np.array([f"IF{spw_i}"], dtype=object), chunks=(1,))
        # TOPO Frame -- we are not regrididng to new reference frequency
        meas_freq_ref = da.asarray(np.array([5], dtype=int), chunks=(1,))
        dataset = Dataset({
            "MEAS_FREQ_REF": (("row",), meas_freq_ref),
            "NUM_CHAN": (("row",), dask_num_chan),
            "CHAN_FREQ": (("row", "chan"), dask_chan_freq),
            "CHAN_WIDTH": (("row", "chan"), dask_chan_width),
            "EFFECTIVE_BW": (("row", "chan"), dask_chan_width),
            "RESOLUTION": (("row", "chan"), dask_chan_width),
            "TOTAL_BANDWIDTH": (("row",), da.sum(dask_chan_width, axis=1)),
            "NAME": (("row",), spw_name)
        })

        spw_datasets.append(dataset)

    # For each cartesian product of SPECTRAL_WINDOW and POLARIZATION
    # create a corresponding DATA_DESCRIPTION.
    # Each column has fixed shape so we handle all rows at once
    spw_ids, pol_ids = zip(*product(range(len(num_freq_channels)),
                                    range(len(corr_types))))
    dask_spw_ids = da.asarray(np.asarray(spw_ids, dtype=np.int32))
    dask_pol_ids = da.asarray(np.asarray(pol_ids, dtype=np.int32))
    ddid_datasets.append(Dataset({
        "SPECTRAL_WINDOW_ID": (("row",), dask_spw_ids),
        "POLARIZATION_ID": (("row",), dask_pol_ids),
    }))

    # Now create the associated MS dataset

    # vis_data, baselines = cal_vis.get_all_visibility()
    # vis_array = np.array(vis_data, dtype=np.complex64)
    chunks = {
        "row": (vis_array.shape[0],),
    }
    baselines = np.array(baselines)
    nbl = np.unique(baselines, axis=0).shape[0]
    baseline_lengths = (da.sqrt((antenna_itrf_pos[baselines[:, 0]] -
                                 antenna_itrf_pos[baselines[:, 1]])**2)).compute()
    
    rayleigh_crit = rayleigh_criterion(max_freq=np.max(spw_chan_freqs), 
                                            baseline_lengths=baseline_lengths)

    LOGGER.info(
        f"Appoximate unweighted instrument resolution: {rayleigh_crit * 60.0:.4f} arcmin")

    # will use casacore to generate these later
    if np.array(timestamps).size > 1 and uvw_generator != 'casacore':
        LOGGER.warning(f"You should not use '{uvw_generator}' mode to generate UVW coordinates"
                       f"for multi-timestamp databases. Your UVW coordinates will be wrong")
    if uvw_generator == 'telescope_snapshot':
        bl_pos = np.array(ant_pos)[baselines]
        uu_a, vv_a, ww_a = -(bl_pos[:, 1] - bl_pos[:, 0]).T
        # Use the - sign to get the same orientation as our tart projections.
        uvw_array = np.array([uu_a, vv_a, ww_a]).T
    elif uvw_generator == 'casacore':
        # to be fixed with our fixvis casacore generator at the end
        uvw_array = np.zeros((vis_array.shape[0], 3), dtype=np.float64)
    else:
        raise ValueError(
            'uvw_generator expects either mode "telescope_snapshot" or "casacore"')

    for ddid, (spw_id, pol_id) in enumerate(zip(spw_ids, pol_ids)):
        # Infer row, chan and correlation shape
        row = sum(chunks['row'])
        chan = spw_datasets[spw_id].CHAN_FREQ.shape[1]
        corr = pol_table.datasets[pol_id].CORR_TYPE.shape[1]

        # Create some dask vis data
        dims = ("row", "chan", "corr")
        LOGGER.debug(f"Data size {row} {chan} {corr}")
        LOGGER.info(
            f"Data column size {row * chan * corr * 8 / 1024.0**2:.2f} MiB")

        np_data = np.zeros((row, chan, corr), dtype=np.complex128)
        for i in range(corr):
            np_data[:, :, i] = vis_array.reshape((row, chan))
        np_uvw = uvw_array.reshape((row, 3))

        data_chunks = tuple((chunks['row'], chan, corr))
        dask_data = da.from_array(np_data, chunks=data_chunks)
        flag_categories = da.from_array(0.05*np.ones((row, chan, corr, 1)))
        flag_data = np.zeros((row, chan, corr), dtype=np.bool_)

        uvw_data = da.from_array(np_uvw)
        # Create dask ddid column
        dask_ddid = da.full(row, ddid, chunks=chunks['row'], dtype=np.int32)
        if np_data.shape[0] % len(epoch_s) != 0:
            raise RuntimeError(
                "Expected nrow to be integral multiple of number of time slots")
        if np_data.shape[0] != len(epoch_s) * nbl:
            raise RuntimeError(
                "Some baselines are missing in the data array. Not supported")
        epoch_s_arr = np.array(epoch_s)
        intervals = np.zeros(len(epoch_s_arr), dtype=np.float64)
        # going to assume the correlator integration interval is constant
        if len(intervals) >= 2:
            # TODO: BH we need a better way of storing intervals and exposures - this is something
            # the correlator must tell us -- only it knows this
            # currently there are edge cases where we have disjoint observations
            intervals[...] = np.median(epoch_s_arr[1:] - epoch_s_arr[:-1])
        else:
            # TODO: Fallover what is the default integration interval
            intervals[0] = 1.0
        intervals = intervals.repeat(nbl)
        # TODO: This should really be made better - partial dumps should be
        # downweighted
        exposure = intervals.copy()
        timems = np.repeat(epoch_s, nbl)

        if phase_center_policy == 'instantaneous-zenith':
            # scan number - treat each integration as a scan
            scan = np.arange(len(epoch_s), dtype=int).repeat(
                nbl) + 1  # offset to start at 1, per convention
            # each integration should have its own phase tracking centre
            # to ensure we can rephase them to a common frame in the end
            field_no = scan.copy() - 1  # offset to start at 0 (FK)
        elif (phase_center_policy == 'no-rephase-obs-midpoint' or
              phase_center_policy == 'rephase-obs-midpoint' or
              phase_center_policy == 'rephase-NCP' or
              phase_center_policy == 'rephase-SCP'):
            # user is just going to get a single zenith position at the observation centoid
            scan = np.ones(len(epoch_s), dtype=int).repeat(
                nbl)  # start at 1, per convention
            field_no = np.zeros_like(scan)
        else:
            raise ValueError(f"phase_center_policy must be one of "
                             f"['instantaneous-zenith','rephase-obs-midpoint','no-rephase-obs-midpoint','rephase-SCP','rephase-NCP'] "
                             f"got {phase_center_policy}")

        # apply rephasor if needed
        mean_sidereal_day = 23 + 56 / 60. + 4.0905 / 3600.  # hrs
        # degrees per second at equator
        sidereal_rate = 360. / (mean_sidereal_day * 3600)
        # if we move more than say 5% of the instrument resolution during the observation
        # then warnings must be raised if we're snapping the field centre without phasing
        obs_length = np.max(epoch_s) - np.min(epoch_s)
        snapshot_length_cutoff = rayleigh_crit / sidereal_rate * 0.05

        if phase_center_policy == 'no-rephase-obs-midpoint' and \
           obs_length > snapshot_length_cutoff:
            LOGGER.critical(f"You are choosing to set the field phase direction at the centre point "
                            f"of the observation without rephasing the original zenithal positions. "
                            f"This is not advised and can cause astrometric errors in your image and "
                            f"incorrect UVW coordinates to be written. Do not do this unless your observation "
                            f"is short enough for sources not to move more than a fraction of the instrumental "
                            f"resolution! You are predicted to move about "
                            f"{np.ceil(obs_length / (rayleigh_crit / sidereal_rate) * 100):.0f}% "
                            f"of the instrument resolution during the course of this observation")

        if phase_center_policy == 'rephase-obs-midpoint' or \
           phase_center_policy == 'rephase-NCP' or \
           phase_center_policy == 'rephase-SCP':
            # we must first have accurate uvw coordinates in each different zenith direction
            assert direction.ndim == 3
            assert direction.shape[0] == 1
            assert direction.shape[1] == 2
            zenith_directions = np.array(
                [[phase_j2000.ra.radian, phase_j2000.dec.radian]])
            zenith_directions = zenith_directions.reshape(zenith_directions.shape[1],
                                                          zenith_directions.shape[2]).T.copy()
            if phase_center_policy == 'rephase-obs-midpoint':
                centroid_direction = zenith_directions[zenith_directions.shape[0]//2, :].reshape(
                    1, 2)
            elif phase_center_policy == 'rephase-NCP':
                centroid_direction = np.array(
                    [0, np.deg2rad(+90)]).reshape(1, 2)
            elif phase_center_policy == 'rephase-SCP':
                centroid_direction = np.array(
                    [0, np.deg2rad(-90)]).reshape(1, 2)
            else:
                raise RuntimeError("Invalid rephase option")

            map_row_to_zendir = np.arange(len(epoch_s), dtype=int).repeat(nbl)
            subfields = np.unique(map_row_to_zendir)
            assert zenith_directions.shape[0] == subfields.size
            p = progress(
                "Computing UVW towards original zenith points", max=subfields.size)
            for sfi in subfields:
                selrow = map_row_to_zendir == sfi
                this_phase_dir = zenith_directions[sfi].reshape(1, 2)
                padded_uvw = synthesize_uvw(station_ECEF=antenna_itrf_pos.compute(),
                                            time=timems[selrow],
                                            a1=baselines[:, 0][selrow],
                                            a2=baselines[:, 1][selrow],
                                            phase_ref=this_phase_dir,
                                            ack=False)
                uvw_data[selrow] = dense2sparse_uvw(a1=baselines[:, 0][selrow],
                                                    a2=baselines[:, 1][selrow],
                                                    time=timems[selrow],
                                                    ddid=(
                                                        np.ones(selrow.size)*ddid)[selrow],
                                                    padded_uvw=padded_uvw["UVW"],
                                                    ack=False)
                p.next()
            new_phase_dir = SkyCoord(centroid_direction[0, 0]*u.rad, centroid_direction[0, 1]*u.rad,
                                     frame='icrs')
            new_phase_dir_repr = f"{new_phase_dir.ra.hms[0]:02.0f}h{new_phase_dir.ra.hms[1]:02.0f}m{new_phase_dir.ra.hms[2]:05.2f}s "\
                                 f"{new_phase_dir.dec.dms[0]:02.0f}d{abs(new_phase_dir.dec.dms[1]):02.0f}m{abs(new_phase_dir.dec.dms[2]):05.2f}s"
            LOGGER.info("<Done>")
            LOGGER.info(
                f"Per user request: Rephase all data to {new_phase_dir_repr}")
            rephased_data = da.empty_like(dask_data)

            rephased_data = \
                da.map_blocks(rephase,
                              dask_data,
                              dtype=dask_data.dtype,
                              chunks=dask_data.chunks,
                              # kwargs for rephase
                              freq=spw_chan_freqs[spw_id],
                              pos=np.rad2deg(centroid_direction[0, :]),
                              uvw=uvw_data,
                              refdir=np.rad2deg(zenith_directions),
                              field_ids=map_row_to_zendir)
            dask_data = rephased_data
        else:
            LOGGER.info("No rephasing requested - field centers left as is")

        dataset = Dataset({
            'DATA': (dims, dask_data.conj()),
            'FLAG': (dims, da.from_array(flag_data)),
            'TIME': (("row",), da.from_array(timems)),
            'TIME_CENTROID': ("row", da.from_array(timems)),
            'WEIGHT': (("row", "corr"), da.from_array(0.95*np.ones((row, corr)))),
            'WEIGHT_SPECTRUM': (dims, da.from_array(0.95*np.ones_like(np_data, dtype=np.float64))),
            # BH: conformance issue, see CASA documentation on weighting
            'SIGMA_SPECTRUM': (dims, da.from_array(np.ones_like(np_data, dtype=np.float64)*0.05)),
            'SIGMA': (("row", "corr"), da.from_array(0.05*np.ones((row, corr)))),
            'UVW': (("row", "uvw",), uvw_data),
            'FLAG_CATEGORY': (('row', 'flagcat', 'chan', 'corr'), flag_categories),
            'ANTENNA1': (("row",), da.from_array(baselines[:, 0])),
            'ANTENNA2': (("row",), da.from_array(baselines[:, 1])),
            'FEED1': (("row",), da.from_array(baselines[:, 0])),
            'FEED2': (("row",), da.from_array(baselines[:, 1])),
            'DATA_DESC_ID': (("row",), dask_ddid),
            'PROCESSOR_ID': (("row",), da.from_array(np.zeros(row, dtype=int), chunks=chunks['row'])),
            'FIELD_ID': (("row",), da.from_array(field_no, chunks=chunks['row'])),
            'INTERVAL': (("row",), da.from_array(intervals, chunks=chunks['row'])),
            'EXPOSURE': (("row",), da.from_array(exposure, chunks=chunks['row'])),
            'SCAN_NUMBER': (("row",), da.from_array(scan, chunks=chunks['row'])),
            'ARRAY_ID': (("row",), da.from_array(np.zeros(row, dtype=int), chunks=chunks['row'])),
            'OBSERVATION_ID': (("row",), da.from_array(np.zeros(row, dtype=int), chunks=chunks['row'])),
            'STATE_ID': (("row",), da.from_array(np.zeros(row, dtype=int), chunks=chunks['row'])),
        })
        ms_datasets.append(dataset)

    ms_writes = xds_to_table(ms_datasets, ms_table_name, columns="ALL")
    spw_writes = xds_to_table(spw_datasets, spw_table_name, columns="ALL")
    ddid_writes = xds_to_table(ddid_datasets, ddid_table_name, columns="ALL")

    dask.compute(ms_writes)

    ant_table.write()
    feed_table.write()
    field_table.write()
    pol_table.write()
    obs_table.write()
    src_table.write()

    dask.compute(spw_writes)
    dask.compute(ddid_writes)

    if uvw_generator == 'telescope_snapshot':
        pass
    elif uvw_generator == 'casacore':
        fixms(ms_table_name)
    else:
        raise ValueError(
            'uvw_generator expects either mode "telescope_snapshot" or "casacore"')


def __print_infodict_keys(dico_info, keys, just=25):
    LOGGER.info("Observatory parameters:")
    for k in keys:
        val = dico_info.get(k, "Information Unavailable")
        reprk = str(k).ljust(just, " ")
        LOGGER.info(f"\t{reprk}: {val}")


def ms_from_hdf5(ms_name, h5file, pol2, phase_center_policy, override_telescope_name, uvw_generator="casacore"):
    if pol2:
        pol_feeds = ['RR', 'LL']
    else:
        pol_feeds = ['RR']
    if isinstance(h5file, str):
        h5file = [h5file]
    all_times = []
    all_vis = []
    all_baselines = []
    ant_pos_orig = None
    orig_dico_info = None
    LOGGER.info("Will process HDF5 file: ")
    for h5 in h5file:
        LOGGER.info(f"\t '{h5}'")

    p = progress("Processing HDF database", max=len(h5file))
    for ih5, h5 in enumerate(h5file):
        with h5py.File(h5, "r") as h5f:
            config_string = np.string_(h5f['config'][0]).decode('UTF-8')
            if ih5 == 0:
                LOGGER.debug("config_string = {}".format(config_string))

            config_json = json.loads(config_string)
            config_json['operating_frequency'] = config_json['frequency']
            if ih5 == 0:
                LOGGER.debug(
                    f"config_json = {json.dumps(config_json, indent=4, sort_keys=True)}")
                __print_infodict_keys(config_json,
                                      ["L0_frequency", "bandwidth", "baseband_frequency",
                                       "operating_frequency", "name", "num_antenna",
                                       "sampling_frequency"])
            config = settings.from_json(config_string)
            hdf_baselines = h5f['baselines'][:]
            hdf_phase_elaz = h5f['phase_elaz'][:]

            ant_pos = h5f['antenna_positions'][:]
            if ant_pos_orig is None:
                ant_pos_orig = ant_pos.copy()
            if not np.isclose(ant_pos_orig, ant_pos).all():
                raise RuntimeError("The databases you are trying to concatenate have different antenna layouts. "
                                   "This is not yet supported. You could try running CASA virtualconcat to "
                                   "concatenate such heterogeneous databases")
            if orig_dico_info is None:
                orig_dico_info = config_json
            config_same = True

            for check_key in ["L0_frequency", "bandwidth", "baseband_frequency",
                              "num_antenna", "operating_frequency", "sampling_frequency",
                              "lat", "lon", "alt", "orientation", "axes"]:
                if check_key not in config_json or check_key not in orig_dico_info:
                    raise RuntimeError(
                        f"Key {check_key} missing from database!")
                if isinstance(orig_dico_info[check_key], float):
                    config_same = config_same and \
                        np.isclose(orig_dico_info[check_key],
                                   config_json[check_key])
                elif isinstance(orig_dico_info[check_key], list):
                    config_same = config_same and \
                        np.all(np.array(orig_dico_info[check_key]) == np.array(
                            config_json[check_key]))
                else:
                    config_same = config_same and \
                        orig_dico_info[check_key] == \
                        config_json[check_key]

            if not config_same:
                raise RuntimeError("The databases you are trying to concatenate have different configurations. "
                                   "This is not yet supported. You could try running CASA virtualconcat to "
                                   "concatenate such heterogeneous databases")
            gains = h5f['gains'][:]
            phases = h5f['phases'][:]

            hdf_timestamps = h5f['timestamp']
            timestamps = [dateutil.parser.parse(x) for x in hdf_timestamps]

            hdf_vis = h5f['vis'][:]

            for ts, v in zip(timestamps, hdf_vis):
                vis = Visibility(config=config, timestamp=ts)
                vis.set_visibilities(v=v, b=hdf_baselines.tolist())
                vis.phase_el = hdf_phase_elaz[0]
                vis.phase_az = hdf_phase_elaz[1]

                cal_vis = calibration.CalibratedVisibility(vis)
                cal_vis.set_gain(np.arange(24), gains)
                cal_vis.set_phase_offset(np.arange(24), phases)

                vis_data, baselines = cal_vis.get_all_visibility()
                vis_array = np.array(vis_data, dtype=np.complex64)

                all_vis.append(vis_array)
                for bl in baselines:
                    all_baselines.append(bl)
                all_times.append(ts)
        p.next()

    LOGGER.info("<Done>")
    # finally create concat ms
    all_vis = np.array(all_vis).flatten()
    all_baselines = np.array(all_baselines)
    ms_create(ms_table_name=ms_name,
              info=orig_dico_info,
              ant_pos=ant_pos_orig,
              vis_array=all_vis,
              baselines=all_baselines,
              timestamps=all_times,
              pol_feeds=pol_feeds,
              sources=[],
              phase_center_policy=phase_center_policy,
              override_telescope_name=override_telescope_name,
              uvw_generator=uvw_generator)


def ms_from_json(ms_name, json_filename, pol2, phase_center_policy, override_telescope_name,
                 uvw_generator="casacore", json_data=None):
    # Load data from a JSON file
    if json_filename is not None and json_data is None:
        if isinstance(json_filename, str):
            json_filename = [json_filename]
        json_data = []
        LOGGER.info("Will process JSON file: ")
        for jfi in json_filename:
            LOGGER.info(f"\t '{jfi}'")
            with open(jfi, 'r') as json_file:
                json_data.append(json.load(json_file))
    elif json_filename is None and json_data is not None:
        if not isinstance(json_data, list):
            json_data = [json_data]
    else:
        raise ValueError(
            "Either json_filename or json_data arguments should be given")

    all_times = []
    all_vis = []
    all_sources = []
    all_baselines = []
    ant_pos_orig = None
    orig_dico_info = None
    p = progress("Processing JSON database", max=len(json_data))
    for ijdi, jdi in enumerate(json_data):
        info = jdi['info']
        ant_pos = jdi['ant_pos']
        config = settings.from_api_json(info['info'], ant_pos)
        gains = jdi['gains']['gain']
        phases = jdi['gains']['phase_offset']
        if ant_pos_orig is None:
            ant_pos_orig = ant_pos.copy()
        if not np.isclose(ant_pos_orig, ant_pos).all():
            raise RuntimeError("The databases you are trying to concatenate have different antenna layouts. "
                               "This is not yet supported. You could try running CASA virtualconcat to "
                               "concatenate such heterogeneous databases")
        if orig_dico_info is None:
            orig_dico_info = info["info"]

        config_same = True
        opt_keys = ["lat", "lon", "alt", "orientation", "axes"]
        for check_key in ["L0_frequency", "bandwidth", "baseband_frequency",
                          "num_antenna", "operating_frequency", "sampling_frequency",
                          "lat", "lon", "alt", "orientation", "axes"]:
            if check_key not in info["info"] or check_key not in orig_dico_info:
                if check_key not in opt_keys:
                    raise RuntimeError(
                        f"Key {check_key} missing from database!")
                else:
                    LOGGER.critical(f"Key {check_key} missing from database, but appears to be optional."
                                    f"You may be using old databases!")
                    continue
            if isinstance(orig_dico_info[check_key], float):
                config_same = config_same and \
                    np.isclose(orig_dico_info[check_key],
                               info["info"][check_key])
            elif isinstance(orig_dico_info[check_key], list):
                config_same = config_same and \
                    np.all(np.array(orig_dico_info[check_key]) == np.array(
                        info["info"][check_key]))
            else:
                config_same = config_same and \
                    orig_dico_info[check_key] == \
                    info["info"][check_key]

        if not config_same:
            raise RuntimeError("The databases you are trying to concatenate have different configurations. "
                               "This is not yet supported. You could try running CASA virtualconcat to "
                               "concatenate such heterogeneous databases")

        # Note, these do not contain the conjugate pairs, only v[i,j] (and not v[j,i])
        # TODO deal with multiple observations in the JSON file later.
        for d in jdi['data']:
            vis_json, source_json = d
            cal_vis, timestamp = api_imaging.vis_calibrated(
                vis_json, config, gains, phases, [])
            src_list = source_json
            # a list of sources per timestamp so we can zip them correctly
            all_sources.append(src_list)
        if pol2:
            pol_feeds = ['RR', 'LL']
        else:
            pol_feeds = ['RR']

        vis_data, baselines = cal_vis.get_all_visibility()
        vis_array = np.array(vis_data, dtype=np.complex64)
        all_vis.append(vis_array)
        for bl in baselines:
            all_baselines.append(bl)
        all_times.append(timestamp)
        p.next()
    LOGGER.info("<Done>")
    __print_infodict_keys(orig_dico_info,
                          ["L0_frequency", "bandwidth", "baseband_frequency",
                           "operating_frequency", "name", "num_antenna",
                           "sampling_frequency"])
    # finally concat into a single measurement set
    all_vis = np.array(all_vis).flatten()
    all_baselines = np.array(all_baselines)
    ms_create(ms_table_name=ms_name,
              info=orig_dico_info,
              ant_pos=ant_pos,
              vis_array=all_vis,
              baselines=all_baselines,
              timestamps=all_times,
              pol_feeds=pol_feeds,
              sources=all_sources,
              phase_center_policy=phase_center_policy,
              override_telescope_name=override_telescope_name, uvw_generator=uvw_generator)
