'''
    Test reading of data from a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2023.

    License. GPLv3.
'''

import unittest

from tart2ms import read_ms, casa_read_ms

TEST_MS = 'test_data/test.ms'


class TestReadMS(unittest.TestCase):

    def array_equal(self, arr_a, arr_b):
        for i in range(arr_a.shape[0]):
            a = arr_a[i]
            b = arr_b[i]
            self.assertAlmostEqual(a, b, 6)

    def test_read_compare(self):
        u_arr2, v_arr2, w_arr2, frequency2, cv_vis2, hdr2, timestamp2, rms_arr2, indices2 = \
            casa_read_ms(ms_file=TEST_MS,
                         num_vis=552,
                         angular_resolution=2.0,
                         channel=0, snapshot=0,
                         field_id=1, pol=0)
        u_arr, v_arr, w_arr, frequency, cv_vis, hdr, timestamp, rms_arr, indices = \
            read_ms(TEST_MS,
                    num_vis=552,
                    angular_resolution=2.0,
                    field_id=1)

        self.array_equal(u_arr, u_arr2)
        self.array_equal(v_arr, v_arr2)
        self.array_equal(w_arr, w_arr2)

        self.array_equal(cv_vis, cv_vis2)

        self.assertEqual(timestamp, timestamp2)
