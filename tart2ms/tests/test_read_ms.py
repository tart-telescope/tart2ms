import unittest

from tart2ms import ms_from_json, ms_from_hdf5, read_ms, casa_read_ms

TEST_MS = 'test_data/test.ms'

class TestReadMS(unittest.TestCase):



    def test_read_compare(self):
        u_arr2, v_arr2, w_arr, frequency, cv_vis, hdr, timestamp, rms_arr, indices = \
            casa_read_ms(ms_file=TEST_MS,
                    num_vis=552,
                    angular_resolution=2.0,
                    channel = 0,
                    snapshot = 0,
                    field_id = 1,
                    pol = 0)
        u_arr, v_arr, w_arr, frequency, cv_vis, hdr, timestamp, rms_arr, indices = \
            read_ms(TEST_MS,
                    num_vis=552,
                    angular_resolution=2.0,
                    field_id = 1)

        for i in range(u_arr.shape[0]):
            a = u_arr[i]
            b = u_arr2[i]
            self.assertAlmostEqual(a, b, 6)

            a = v_arr[i]
            b = v_arr2[i]
            self.assertAlmostEqual(a, b, 6)
