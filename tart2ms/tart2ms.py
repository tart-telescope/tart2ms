'''
    A quick attempt to get TART JSON data into a measurement set.
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

from casacore.measures import measures

from astropy import coordinates as ac

from daskms import Dataset, xds_to_table

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy.utils import iers
from astropy.constants import R_earth

from tart.util import constants
from tart.operation import settings
from tart.imaging.visibility import Visibility
from tart.imaging import calibration

from tart_tools import api_imaging
from .fixvis import fixms

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
    dm = measures()
    epoch = dm.epoch(rf='utc', v0=t_stamp.isoformat())
    epoch_d = epoch['m0']['value']
    epoch_s = epoch_d*24*60*60.0
    return epoch_s


def ms_create(ms_table_name, info, ant_pos, vis_array, baselines, timestamps, pol_feeds, sources, phase_center_policy, override_telescope_name):
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
    except:
        loc = info

    lat, lon, height = loc["lat"], loc["lon"], loc["alt"]
    LOGGER.info(f"pos {lat, lon, height}")

    array_centroid = ac.EarthLocation.from_geodetic(lat=lat, lon=lon, height=height)
    epoch_s = list(map(timestamp_to_ms_epoch, timestamps))
    LOGGER.info(f"Time {epoch_s}")

    # Sort out the coordinate frames using astropy
    # https://casa.nrao.edu/casadocs/casa-5.4.1/reference-material/coordinate-frames
    iers.conf.iers_auto_url = 'https://astroconda.org/aux/astropy_mirror/iers_a_1/finals2000A.all'
    iers.conf.auto_max_age = None

    location = EarthLocation.from_geodetic(lon=loc['lon']*u.deg,
                                           lat=loc['lat']*u.deg,
                                           height=loc['alt']*u.m,
                                           ellipsoid='WGS84')
    obstime = Time(timestamps)
    LOGGER.debug(f"obstime {obstime}")

    # local_frame = AltAz(obstime=obstime, location=location)
    # LOGGER.info(f"local_frame {local_frame}")

    phase_altaz = SkyCoord(alt=[90.0*u.deg]*len(obstime), az=[0.0*u.deg]*len(obstime),
                           obstime = obstime, frame = 'altaz', location = location)
    phase_j2000 = phase_altaz.transform_to('fk5')
    LOGGER.debug(f"phase_j2000 {phase_j2000}")

    # Get the stokes enums for the polarization types
    corr_types = [[MS_STOKES_ENUMS[p_f] for p_f in pol_feeds]]

    LOGGER.info("Pol Feeds {}".format(pol_feeds))
    LOGGER.info("Correlation Types {}".format(corr_types))
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
    
    ########################### Now convert each antenna location to ECEF coordinates for the measurement set. #######################
    # This is laborious but seems to work.
    #
    ant_posang = [Angle(np.arctan2(a[0],a[1]), unit=u.rad) for a in ant_pos]  # Zero is due north.
    ant_s = [np.sqrt(a[0]*a[0] + a[1]*a[1])  for a in ant_pos]
    ant_distance = [s / R_earth.value  for s in ant_s]
    
    ant_lon_lat = [ac.offset_by(lon=lon*u.deg, lat=lat*u.deg, posang=theta, distance=d)  for theta, d in zip(ant_posang, ant_distance)]
    ant_locations = [EarthLocation.from_geodetic(lon=lon,  lat=lat, height=loc['alt']*u.m,  ellipsoid='WGS84') for lon, lat in ant_lon_lat]
    ant_positions = [[e.x.value, e.y.value, e.z.value] for e in ant_locations]
    position = da.asarray(ant_positions)
    
    
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

    ###################  Create a FEED dataset. ###################################
    # There is one feed per antenna, so this should be quite similar to the ANTENNA
    num_pols = len(pol_feeds)
    pol_types = pol_feeds
    pol_responses = [POL_RESPONSES[ct] for ct in pol_feeds]

    LOGGER.info("Pol Types {}".format(pol_types))
    LOGGER.info("Pol Responses {}".format(pol_responses))

    antenna_ids = da.asarray(range(num_ant))
    feed_ids = da.zeros(num_ant)
    num_receptors = da.zeros(num_ant) + num_pols
    polarization_types = np.array([pol_types for i in range(num_ant)], dtype=object)
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


    ####################### FIELD dataset ####################################

    direction = np.array([[phase_j2000.ra.radian, phase_j2000.dec.radian]])
    assert direction.ndim == 3
    assert direction.shape[0] == 1
    assert direction.shape[1] == 2
    if phase_center_policy == "dump":
        pass
    elif phase_center_policy == "observation":
        direction = direction[:,:,direction.shape[2]//2].reshape(1,2,1)
    else:
        raise ValueError(f"phase_center_policy must be one of [dump, observation] got {phase_center_policy}")
    field_direction = da.asarray(
            direction.T.reshape(direction.shape[2],
                                1, 2).copy(), chunks=(1, None, None)) # nrow x npoly x 2
    field_name = da.asarray(np.array(list(map(lambda sn: f'zenith_scan_{sn+1}', range(direction.shape[2]))),
                                     dtype=object),
                            chunks=1)
    field_num_poly = da.zeros(direction.shape[2], chunks=1) # zeroth order polynomial in time for phase center.
    dir_dims = ("row", 'field-poly', 'field-dir',)

    dataset = Dataset({
        'PHASE_DIR': (dir_dims, field_direction),
        'DELAY_DIR': (dir_dims, field_direction),
        'REFERENCE_DIR': (dir_dims, field_direction),
        'NUM_POLY': (("row", ), field_num_poly),
        'NAME': (("row", ), field_name),
    })
    field_table.append(dataset)

   ######################### OBSERVATION dataset #####################################
    LOGGER.info(f"Writing MS for telescope name '{override_telescope_name}'")
    dataset = Dataset({
        'TELESCOPE_NAME': (("row",), da.asarray(np.asarray([override_telescope_name], dtype=object), chunks=1)),
        'OBSERVER': (("row",), da.asarray(np.asarray(['Tim'], dtype=object), chunks=1)),
        "TIME_RANGE": (("row","obs-exts"), da.asarray(np.array([[epoch_s[0], epoch_s[-1]]]), chunks=1)),
    })
    obs_table.append(dataset)

    ######################## SOURCE datasets ########################################
    for src in sources:
        name = src['name']
        # Convert to J2000
        dir_altaz = SkyCoord(alt=src['el']*u.deg, az=src['az']*u.deg, obstime = obstime[0],
                             frame = 'altaz', location = location)
        dir_j2000 = dir_altaz.transform_to('fk5')
        direction = [dir_j2000.ra.radian, dir_j2000.dec.radian]
        LOGGER.info(f"SOURCE: {name}, timestamp: {timestamps}, dir: {direction}")
        dask_num_lines = da.asarray(np.asarray([1], dtype=np.int32)) # , 1, dtype=np.int32)
        dask_direction = da.asarray(np.asarray(direction, dtype=np.float64), chunks=1)[None, :]
        dask_name = da.asarray(np.asarray([name], dtype=object), chunks=1)
        dask_time = da.asarray(np.asarray(epoch_s, dtype=object), chunks=1)
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
        LOGGER.info("Corr Prod {}".format(corr_prod))
        LOGGER.info("Corr Type {}".format(corr_type))

        dask_num_corr = da.full((1,), len(corr_type), dtype=np.int32)
        LOGGER.info("NUM_CORR {}".format(dask_num_corr))
        dask_corr_type = da.from_array(corr_type,
                                       chunks=len(corr_type))[None, :]
        dask_corr_product = da.asarray(corr_prod)[None, :]
        LOGGER.info("Dask Corr Prod {}".format(dask_corr_product.shape))
        LOGGER.info("Dask Corr Type {}".format(dask_corr_type.shape))
        dataset = Dataset({
            "NUM_CORR": (("row",), dask_num_corr),
            "CORR_TYPE": (("row", "corr"), dask_corr_type),
            "CORR_PRODUCT": (("row", "corr", "corrprod_idx"), dask_corr_product),
        })

        pol_table.append(dataset)

    # Create multiple SPECTRAL_WINDOW datasets
    # Dataset per output row required because column shapes are variable

    for spw_i, num_chan in enumerate(num_freq_channels):
        dask_num_chan = da.full((1,), num_chan, dtype=np.int32, chunks=(1,))
        dask_chan_freq = da.asarray([[info['operating_frequency']]], chunks=(1, None))
        dask_chan_width = da.full((1, num_chan), 2.5e6/num_chan, chunks=(1, None))
        spw_name = da.asarray(np.array([f"IF{spw_i}"], dtype=object), chunks=(1,))
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

    #vis_data, baselines = cal_vis.get_all_visibility()
    #vis_array = np.array(vis_data, dtype=np.complex64)
    chunks = {
        "row": (vis_array.shape[0],),
    }
    baselines = np.array(baselines)
    nbl = np.unique(baselines, axis=0).shape[0]
    
    # will use casacore to generate these later
    uvw_array = np.zeros((vis_array.shape[0], 3), dtype=np.float64)
    
    for ddid, (spw_id, pol_id) in enumerate(zip(spw_ids, pol_ids)):
        # Infer row, chan and correlation shape
        #LOGGER.info("ddid:{} ({}, {})".format(ddid, spw_id, pol_id))
        row = sum(chunks['row'])
        chan = spw_datasets[spw_id].CHAN_FREQ.shape[1]
        corr = pol_table.datasets[pol_id].CORR_TYPE.shape[1]

        # Create some dask vis data
        dims = ("row", "chan", "corr")
        LOGGER.info(f"Data size {row} {chan} {corr}")

        #np_data = vis_array.reshape((row, chan, corr))
        np_data = np.zeros((row, chan, corr), dtype=np.complex128)
        for i in range(corr):
            np_data[:, :, i] = vis_array.reshape((row, chan))
        #np_data = np.array([vis_array.reshape((row, chan, 1)) for i in range(corr)])
        np_uvw = uvw_array.reshape((row, 3))

        data_chunks = tuple((chunks['row'], chan, corr))
        dask_data = da.from_array(np_data, chunks=data_chunks)

        flag_categories = da.from_array(0.05*np.ones((row, chan, corr, 1)))
        flag_data = np.zeros((row, chan, corr), dtype=np.bool_)

        uvw_data = da.from_array(np_uvw)
        # Create dask ddid column
        dask_ddid = da.full(row, ddid, chunks=chunks['row'], dtype=np.int32)
        assert dask_data.shape[0] % len(epoch_s) == 0, "Expected nrow to be divisible by ntime"
        assert dask_data.shape[0] == len(epoch_s) * nbl
        epoch_s_arr = np.array(epoch_s)
        intervals = np.zeros(len(epoch_s_arr), dtype=np.float64)
        # going to assume the correlator integration interval is constant
        intervals[:-1] = epoch_s_arr[1:] - epoch_s_arr[:-1]
        if len(intervals) >= 2:
            intervals[-1] = intervals[-2]
        else:
            intervals[0] = 1.0 # TODO: Fallover what is the default integration interval
        assert all(np.abs(intervals[1:] - intervals[:-1]) < 0.1), "Correlator dump interval must be regular"
        intervals = intervals.repeat(nbl)
        # TODO: This should really be made better - partial dumps should be
        # downweighted
        exposure = intervals.copy()
        
        if phase_center_policy == "dump":
            # scan number - treat each integration as a scan
            scan = np.arange(len(epoch_s), dtype=int).repeat(nbl) + 1 # offset to start at 1, per convention
            # each integration should have its own phase tracking centre
            # to ensure we can rephase them to a common frame in the end
            field_no = scan.copy() - 1 # offset to start at 0 (FK)
        elif phase_center_policy == "observation":
            # user is just going to get a single zenith position at the observation centoid
            scan = np.ones(len(epoch_s), dtype=int).repeat(nbl) # start at 1, per convention
            field_no = np.zeros_like(scan)
        else:
            raise ValueError(f"phase_center_policy must be one of [dump, observation] got {phase_center_policy}")
        
        dataset = Dataset({
            'DATA': (dims, dask_data),
            'FLAG': (dims, da.from_array(flag_data)),
            'TIME': (("row",), da.from_array(np.repeat(epoch_s, nbl))),
            'TIME_CENTROID': ("row", da.from_array(np.repeat(epoch_s, nbl))),
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


def ms_from_hdf5(ms_name, h5file, pol2, phase_center_policy, override_telescope_name):
    LOGGER.info(f"Dumping phase center per {phase_center_policy}")
    if pol2:
        pol_feeds = [ 'RR', 'LL' ]
    else:
        pol_feeds = [ 'RR' ]
    if isinstance(h5file, str):
        h5file = [h5file]
    all_times = []
    all_vis = []
    all_baselines = []
    ant_pos_orig = None
    orig_dico_info = None

    for ih5, h5 in enumerate(h5file):
        with h5py.File(h5, "r") as h5f:
            LOGGER.info(f"Processing h5 database {ih5+1}/{len(h5file)}: '{h5}'")
            config_string = np.string_(h5f['config'][0]).decode('UTF-8')
            if ih5 == 0:
                LOGGER.info("config_string = {}".format(config_string))

            config_json = json.loads(config_string)
            config_json['operating_frequency'] = config_json['frequency']
            if ih5 == 0:
                LOGGER.info(f"config_json = {json.dumps(config_json, indent=4, sort_keys=True)}")
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
            print(config_json)
            for check_key in ["L0_frequency", "bandwidth", "baseband_frequency",
                              "num_antenna", "operating_frequency", "sampling_frequency",
                              "lat", "lon", "alt", "orientation", "axes"]:
                if check_key not in config_json or check_key not in orig_dico_info:
                    raise RuntimeError(f"Key {check_key} missing from database!")
                if isinstance(orig_dico_info[check_key], float):
                    config_same = np.isclose(orig_dico_info[check_key],
                                             config_json[check_key])
                elif isinstance(orig_dico_info, list):
                    config_same = all(orig_dico_info, config_json)
                else:
                    config_same = orig_dico_info[check_key] == \
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

    # finally create concat ms
    all_vis = np.array(all_vis).flatten()
    all_baselines = np.array(all_baselines)
    ms_create(ms_table_name=ms_name,
                info = orig_dico_info,
                ant_pos = ant_pos_orig,
                vis_array = all_vis,
                baselines= all_baselines,
                timestamps= all_times,
                pol_feeds= pol_feeds,
                sources=[],
                phase_center_policy=phase_center_policy,
                override_telescope_name=override_telescope_name)
    fixms(ms_name)

def ms_from_json(ms_name, json_data, pol2, phase_center_policy, override_telescope_name):
    LOGGER.info(f"Dumping phase center per {phase_center_policy}")
    info = json_data['info']
    ant_pos = json_data['ant_pos']
    config = settings.from_api_json(info['info'], ant_pos)
    gains = json_data['gains']['gain']
    phases = json_data['gains']['phase_offset']

    # Note, these do not contain the conjugate pairs, only v[i,j] (and not v[j,i])
    for d in json_data['data']: # TODO deal with multiple observations in the JSON file later.
        vis_json, source_json = d
        cal_vis, timestamp = api_imaging.vis_calibrated(vis_json, config, gains, phases, [])
        src_list = source_json

    if pol2:
        pol_feeds = [ 'RR', 'LL' ]
    else:
        pol_feeds = [ 'RR' ]

    vis_data, baselines = cal_vis.get_all_visibility() 
    vis_array = np.array(vis_data, dtype=np.complex64)

    ms_create(ms_table_name=ms_name, info = info['info'],
              ant_pos = ant_pos,
              vis_array = vis_array, baselines=baselines, timestamps=[timestamp],
              pol_feeds=pol_feeds, sources=src_list, phase_center_policy=phase_center_policy,
              override_telescope_name=override_telescope_name)
    fixms(ms_name)