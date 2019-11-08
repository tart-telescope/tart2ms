#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import json
import shutil
import os
import tempfile

import numpy as np

from tart2ms import ms_from_json

TEST_JSON = 'tart2ms/tests/data_test.json'
TEST_MS = os.path.join(tempfile.gettempdir(), 'test.ms')

class TestTart2MS(unittest.TestCase):

    def setUp(self):
        with open(TEST_JSON, 'r') as json_file:
            self.json_data = json.load(json_file)
            
    def test_local_dir(self):
        shutil.rmtree(TEST_MS, ignore_errors=True)
        ms_from_json(TEST_MS, self.json_data, pol2=False)
        self.assertTrue(os.path.exists(TEST_MS))
        shutil.rmtree(TEST_MS)

    def test_tmp_dir(self):
        shutil.rmtree(TEST_MS, ignore_errors=True)
        ms_from_json(TEST_MS, self.json_data, pol2=False)
        self.assertTrue(os.path.exists(TEST_MS))
        shutil.rmtree(TEST_MS)

