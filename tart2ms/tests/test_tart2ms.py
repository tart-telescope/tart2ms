#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import json
import shutil
import os
import tempfile
import logging

import numpy as np
from tart.operation import settings
from tart_tools import api_imaging

from tart2ms import ms_from_json, ms_from_hdf5, read_ms

AFRICANUS_DFT_AVAIL = True
try:
    from africanus.rime.dask import wsclean_predict
    from africanus.coordinates.dask import radec_to_lm
except ImportError:
    AFRICANUS_DFT_AVAIL = False

TEST_H5 = './test_data/vis_2021-03-25_20_50_23.568474.hdf'
TEST_JSON = 'tart2ms/tests/data_test.json'
TMP_MS = os.path.join(tempfile.gettempdir(), 'test.ms')
TEST_MS = 'test.ms'

logger = logging.getLogger("tart2ms")
# Add a null handler so logs can go somewhere
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


class TestTart2MS(unittest.TestCase):

    def check_ms(self, test_ms):
        shutil.rmtree(test_ms, ignore_errors=True)
        ms_from_json(test_ms, TEST_JSON, pol2=False,
                     phase_center_policy='instantaneous-zenith',
                     override_telescope_name='TART',
                     uvw_generator="telescope_snapshot",
                     fetch_sources=False)

    def setUp(self):
        with open(TEST_JSON, 'r') as json_file:
            self.json_data = json.load(json_file)

    def test_local_dir(self):
        self.check_ms('test.ms')
        self.assertTrue(os.path.exists('test.ms'))
        shutil.rmtree('test.ms')

    def test_tmp_dir(self):
        self.check_ms(TMP_MS)
        self.assertTrue(os.path.exists(TMP_MS))
        shutil.rmtree(TMP_MS)

    def test_uv_equal(self):
        self.check_ms(TEST_MS)

        res_deg = 2.0

        u_arr, v_arr, w_arr, frequency, cv_vis, hdr, timestamp, rms_arr, indices = read_ms(TEST_MS,
                                                                                           num_vis=552,
                                                                                           angular_resolution=res_deg)
        logger.info("U shape: {}".format(u_arr.shape))

        info = self.json_data['info']
        ant_pos = self.json_data['ant_pos']
        config = settings.from_api_json(info['info'], ant_pos)

        gains_json = self.json_data['gains']
        gains = np.asarray(gains_json['gain'])
        phase_offsets = np.asarray(gains_json['phase_offset'])

        cal_vis, timestamp = api_imaging.vis_calibrated(self.json_data['data'][0][0],
                                                        config, gains, phase_offsets, [])
        # c = cal_vis.get_config()

        # We need to get the vis array to be correct for the full set of u,v,w points (baselines),
        # including the -u,-v, -w points.

        u_arr2, v_arr2, w_arr2 = cal_vis.get_all_uvw()

        self.assertAlmostEqual(np.max(u_arr, axis=0),
                               np.max(u_arr2, axis=0), 6)

        logger.info("U2 shape {}".format(u_arr2.shape))

        for i in range(u_arr.shape[0]):
            a = u_arr[i]
            b = u_arr2[i]
            self.assertAlmostEqual(a, b, 6)

            a = v_arr[i]
            b = v_arr2[i]
            self.assertAlmostEqual(a, b, 6)

    def test_from_h5(self):
        '''
            Use an h5 file and read it in...
        '''

        ms_from_hdf5(ms_name='test_h5.ms', h5file=TEST_H5, pol2=False,
                     phase_center_policy='instantaneous-zenith',
                     override_telescope_name='TART',
                     uvw_generator="telescope_snapshot",
                     fetch_sources=False)

        self.assertTrue(True)

    def test_model_predict(self, test_ms="test_json_with_model.ms"):
        if AFRICANUS_DFT_AVAIL:
            shutil.rmtree(test_ms, ignore_errors=True)
            ms_from_json(test_ms, TEST_JSON, pol2=False,
                        phase_center_policy='instantaneous-zenith',
                        override_telescope_name='TART',
                        uvw_generator="telescope_snapshot",
                        fill_model=True,
                        writemodelcatalog=True,
                        fetch_sources=False)
            from pyrap.tables import table as tbl
            with tbl(test_ms) as tt:
                self.assertTrue("MODEL_DATA" in tt.colnames())
        else:
            pass
