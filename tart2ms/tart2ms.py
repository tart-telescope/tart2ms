'''
    A quick attempt to get TART JSON data into a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019.

    Official Documentation is Here.
        https://casa.nrao.edu/Memos/229.html#SECTION00044000000000000000

    License. GPLv3.
'''
import logging
from itertools import product
import h5py
import dateutil
import json

import dask
import dask.array as da

import numpy as np

from casacore.measures import measures

from daskms import Dataset, xds_to_table

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.utils import iers

from tart.util import constants
from tart.operation import settings
from tart_tools import api_imaging
from tart.imaging.visibility import Visibility
from tart.imaging import calibration

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


def timestamp_to_ms_epoch(ts):
    ''' Convert an timestamp to seconds (epoch values)
        epoch suitable for using in a Measurement Set
        
    Parameters
    ----------
    ts : A timestamp object.

    Returns
    -------
    t : float
      The epoch time ``t`` in seconds suitable for fields in 
      measurement sets.
    '''
    dm = measures()
    epoch = dm.epoch(rf='utc', v0=ts.isoformat())
    epoch_d = epoch['m0']['value']
    epoch_s = epoch_d*24*60*60.0
    return epoch_s


def ms_create(ms_table_name, info, ant_pos, vis_array, baselines, timestamps, pol_feeds, sources):
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

    epoch_s = timestamp_to_ms_epoch(timestamps)
    LOGGER.info("Time {}".format(epoch_s))

    try:
        loc = info['location']
    except:
        loc = info
    # Sort out the coordinate frames using astropy
    # https://casa.nrao.edu/casadocs/casa-5.4.1/reference-material/coordinate-frames
    iers.conf.iers_auto_url = 'https://astroconda.org/aux/astropy_mirror/iers_a_1/finals2000A.all' 
    iers.conf.auto_max_age = None 

    location = EarthLocation.from_geodetic(lon=loc['lon']*u.deg,
                                           lat=loc['lat']*u.deg,
                                           height=loc['alt']*u.m,
                                           ellipsoid='WGS84')
    obstime = Time(timestamps)
    local_frame = AltAz(obstime=obstime, location=location)

    phase_altaz = SkyCoord(alt=90.0*u.deg, az=0.0*u.deg, obstime = obstime, frame = 'altaz', location = location)
    phase_j2000 = phase_altaz.transform_to('fk5')

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
    position = da.asarray(ant_pos)
    diameter = da.ones(num_ant) * 0.025
    offset = da.zeros((num_ant, 3))
    names = np.array(['ANTENNA-%d' % i for i in range(num_ant)], dtype=np.object)
    stations = np.array([info['name'] for i in range(num_ant)], dtype=np.object)

    dataset = Dataset({
        'POSITION': (("row", "xyz"), position),
        'OFFSET': (("row", "xyz"), offset),
        'DISH_DIAMETER': (("row",), diameter),
        'NAME': (("row",), da.from_array(names, chunks=num_ant)),
        'STATION': (("row",), da.from_array(stations, chunks=num_ant)),
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
    polarization_types = np.array([pol_types for i in range(num_ant)], dtype=np.object)
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


    ####################### FIELD dataset #########################################
    
    direction = [[phase_j2000.ra.radian, phase_j2000.dec.radian]]
    field_direction = da.asarray(direction)[None, :]
    field_name = da.asarray(np.asarray(['up'], dtype=np.object), chunks=1)
    field_num_poly = da.zeros(1) # Zero order polynomial in time for phase center.

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

    dataset = Dataset({
        'TELESCOPE_NAME': (("row",), da.asarray(np.asarray(['TART'], dtype=np.object), chunks=1)),
        'OBSERVER': (("row",), da.asarray(np.asarray(['Tim'], dtype=np.object), chunks=1)),
        "TIME_RANGE": (("row","obs-exts"), da.asarray(np.array([[epoch_s, epoch_s+1]]), chunks=1)),
    })
    obs_table.append(dataset)

    ######################## SOURCE datasets ########################################
    for src in sources:
        name = src['name']
        # Convert to J2000 
        dir_altaz = SkyCoord(alt=src['el']*u.deg, az=src['az']*u.deg, obstime = obstime,
                             frame = 'altaz', location = location)
        dir_j2000 = dir_altaz.transform_to('fk5')
        direction = [dir_j2000.ra.radian, dir_j2000.dec.radian]
        #LOGGER.info("SOURCE: {}, timestamp: {}".format(name, timestamps))
        dask_num_lines = da.full((1,), 1, dtype=np.int32)
        dask_direction = da.asarray(direction)[None, :]
        dask_name = da.asarray(np.asarray([name], dtype=np.object), chunks=1)
        dask_time = da.asarray(np.array([epoch_s]))
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

    for num_chan in num_freq_channels:
        dask_num_chan = da.full((1,), num_chan, dtype=np.int32)
        dask_chan_freq = da.asarray([[info['operating_frequency']]])
        dask_chan_width = da.full((1, num_chan), 2.5e6/num_chan)

        dataset = Dataset({
            "NUM_CHAN": (("row",), dask_num_chan),
            "CHAN_FREQ": (("row", "chan"), dask_chan_freq),
            "CHAN_WIDTH": (("row", "chan"), dask_chan_width),
            "EFFECTIVE_BW": (("row", "chan"), dask_chan_width),
            "RESOLUTION": (("row", "chan"), dask_chan_width),
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
    #LOGGER.info(f"baselines {baselines}")
    bl_pos = np.array(ant_pos)[baselines]
    uu_a, vv_a, ww_a = -(bl_pos[:, 1] - bl_pos[:, 0]).T #/constants.L1_WAVELENGTH
    # Use the - sign to get the same orientation as our tart projections.

    uvw_array = np.array([uu_a, vv_a, ww_a]).T

    for ddid, (spw_id, pol_id) in enumerate(zip(spw_ids, pol_ids)):
        # Infer row, chan and correlation shape
        #LOGGER.info("ddid:{} ({}, {})".format(ddid, spw_id, pol_id))
        row = sum(chunks['row'])
        chan = spw_datasets[spw_id].CHAN_FREQ.shape[1]
        corr = pol_table.datasets[pol_id].CORR_TYPE.shape[1]

        # Create some dask vis data
        dims = ("row", "chan", "corr")
        LOGGER.info("Data size %s %s %s" % (row, chan, corr))

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
        dataset = Dataset({
            'DATA': (dims, dask_data),
            'FLAG': (dims, da.from_array(flag_data)),
            'TIME': (("row", "corr"), da.from_array(epoch_s*np.ones((row, corr)))),
            'TIME_CENTROID': (("row", "corr"), da.from_array(epoch_s*np.ones((row, corr)))),
            'WEIGHT': (("row", "corr"), da.from_array(0.95*np.ones((row, corr)))),
            'WEIGHT_SPECTRUM': (dims, da.from_array(0.95*np.ones_like(np_data, dtype=np.float64))),
            'SIGMA_SPECTRUM': (dims, da.from_array(np.ones_like(np_data, dtype=np.float64)*0.05)),
            'SIGMA': (("row", "corr"), da.from_array(0.05*np.ones((row, corr)))),
            'UVW': (("row", "uvw",), uvw_data),
            'FLAG_CATEGORY': (('row', 'flagcat', 'chan', 'corr'), flag_categories), # {'dims': ('flagcat', 'chan', 'corr')}
            'ANTENNA1': (("row",), da.from_array(baselines[:, 0])),
            'ANTENNA2': (("row",), da.from_array(baselines[:, 1])),
            'FEED1': (("row",), da.from_array(baselines[:, 0])),
            'FEED2': (("row",), da.from_array(baselines[:, 1])),
            'DATA_DESC_ID': (("row",), dask_ddid)
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


def ms_from_hdf5(ms_name, h5file, pol2):
    if pol2:
        pol_feeds = [ 'RR', 'LL' ]
    else:
        pol_feeds = [ 'RR' ]

    with h5py.File(h5file, "r") as h5f:
        config_string = np.string_(h5f['config'][0]).decode('UTF-8')
        LOGGER.info("config_string = {}".format(config_string))
        
        config_json = json.loads(config_string)
        config_json['operating_frequency'] = config_json['frequency']
        
        LOGGER.info("config_json = {}".format(json.dumps(config_json, indent=4, sort_keys=True)))
        config = settings.from_json(config_string)
        hdf_baselines = h5f['baselines'][:]
        hdf_phase_elaz = h5f['phase_elaz'][:]

        ant_pos = h5f['antenna_positions'][:]
        
        gains = h5f['gains'][:]
        phases = h5f['phases'][:]

        hdf_timestamps = h5f['timestamp']
        timestamps = [dateutil.parser.parse(x) for x in hdf_timestamps]
        
        hdf_vis = h5f['vis'][:]
    
        all_times = []
        all_vis = []
        all_baselines = []

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

            #name = "{}_{}.ms".format(ms_name, ts)
            
            #import re
            #name = re.sub(r'[^\w_\.]', '_', name)
    
        all_vis = np.array(all_vis).flatten()
        all_baselines = np.array(all_baselines)
        ms_create(ms_table_name=ms_name, info = config_json,
                    ant_pos = ant_pos,
                    vis_array = all_vis, baselines=all_baselines, timestamps=ts,
                    pol_feeds=pol_feeds, sources=[])


def ms_from_json(ms_name, json_data, pol2):
    info = json_data['info']
    ant_pos = json_data['ant_pos']
    config = settings.from_api_json(info['info'], ant_pos)
    gains = json_data['gains']['gain']
    phases = json_data['gains']['phase_offset']

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
              vis_array = vis_array, baselines=baselines, timestamps=timestamp,
              pol_feeds=pol_feeds, sources=src_list)
