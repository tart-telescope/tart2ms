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

from tart2ms import ms_from_json
import disko

TEST_JSON = 'tart2ms/tests/data_test.json'
TEST_MS = os.path.join(tempfile.gettempdir(), 'test.ms')

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

class TestTart2MS(unittest.TestCase):

    def setUp(self):
        with open(TEST_JSON, 'r') as json_file:
            self.json_data = json.load(json_file)
            
    def test_local_dir(self):
        shutil.rmtree('test.ms', ignore_errors=True)
        ms_from_json('test.ms', self.json_data, pol2=False)
        self.assertTrue(os.path.exists('test.ms'))
        shutil.rmtree('test.ms')

    def test_tmp_dir(self):
        shutil.rmtree(TEST_MS, ignore_errors=True)
        ms_from_json(TEST_MS, self.json_data, pol2=False)
        self.assertTrue(os.path.exists(TEST_MS))
        shutil.rmtree(TEST_MS)


    def test_uv_equal(self):
        shutil.rmtree(TEST_MS, ignore_errors=True)
        ms_from_json(TEST_MS, self.json_data, pol2=False)

        res = disko.Resolution.from_deg(2)

        u_arr, v_arr, w_arr, frequency, cv_vis, hdr, timestamp, rms_arr, indices = disko.read_ms(TEST_MS,
                                                                         num_vis=276, 
                                                                         resolution=res)
        logger.info("U shape: {}".format(u_arr.shape))
        
        info = self.json_data['info']
        ant_pos = self.json_data['ant_pos']
        config = settings.from_api_json(info['info'], ant_pos)
        
        gains_json = self.json_data['gains']
        gains = np.asarray(gains_json['gain'])
        phase_offsets = np.asarray(gains_json['phase_offset'])


        cal_vis, timestamp = api_imaging.vis_calibrated(self.json_data['data'][0][0],
                                                        config, gains, phase_offsets, [])
        c = cal_vis.get_config()
        ant_p = np.asarray(c.get_antenna_positions())

        # We need to get the vis array to be correct for the full set of u,v,w points (baselines), 
        # including the -u,-v, -w points.

        baselines, u_arr2, v_arr2, w_arr2 = disko.get_all_uvw(ant_p)
        
        self.assertAlmostEqual(np.max(u_arr, axis=0), np.max(u_arr2, axis=0), 6)
        
        logger.info("U2 shape {}".format(u_arr2.shape))
        
        for i in range(u_arr.shape[0]):
            a = u_arr[i]
            b = u_arr2[i]
            self.assertAlmostEqual(a,b,6)
            
            a = v_arr[i]
            b = v_arr2[i]
            self.assertAlmostEqual(a,b,6)
            
