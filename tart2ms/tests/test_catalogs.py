import unittest
from tart2ms.catalogs import catalog_reader
from tart2ms import util
from collections import namedtuple

class TestCatalogs(unittest.TestCase):
    def test_3CRR(self):
        self.assertTrue(len(catalog_reader.catalog_factory.from_3CRR()) > 1)
    
    def test_SUMMS(self):
        self.assertTrue(len(catalog_reader.catalog_factory.from_SUMMS(fluxlim15=0.1)) > 1)

    def test_MKGains(self):
        self.assertTrue(len(catalog_reader.catalog_factory.from_MKGains(fluxlim15=0.1)) > 1)
    
    def test_specialpos(self):
        self.assertTrue(len(util.read_known_phasings()) > 1)

    def test_coordreader(self):
        self.assertTrue(util.read_coordinate_twelveball("J193900:632400") == None)
        self.assertTrue(util.read_coordinate_twelveball("J1939004-632400") == None)
        self.assertTrue(util.read_coordinate_twelveball("J193900-6324006") == None)
        self.assertTrue(abs(util.read_coordinate_twelveball("J193945-632423").ra.hms[0] - 19) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945-632423").ra.hms[1] - 39) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945-632423").ra.hms[2] - 45) < 1e-1 and 
                        abs(util.read_coordinate_twelveball("J193945-632423").dec.dms[0] - -63) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945-632423").dec.dms[1] - -24) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945-632423").dec.dms[2] - -23) < 1e-1)
        self.assertTrue(abs(util.read_coordinate_twelveball("J193945+632423").ra.hms[0] - 19) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945+632423").ra.hms[1] - 39) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945+632423").ra.hms[2] - 45) < 1e-1 and 
                        abs(util.read_coordinate_twelveball("J193945+632423").dec.dms[0] - +63) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945+632423").dec.dms[1] - +24) < 1e-10 and 
                        abs(util.read_coordinate_twelveball("J193945+632423").dec.dms[2] - +23) < 1e-1)
        self.assertTrue(util.read_coordinate_twelveball("B193945+632423") is not None)
