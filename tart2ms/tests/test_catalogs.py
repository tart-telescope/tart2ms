import unittest
from tart2ms.catalogs import catalog_reader

class TestCatalogs(unittest.TestCase):
    def test_3CRR(self):
        self.assertTrue(len(catalog_reader.catalog_factory.from_3CRR()) > 1)
    
    def test_SUMMS(self):
        self.assertTrue(len(catalog_reader.catalog_factory.from_SUMMS(fluxlim15=0.1)) > 1)

    def test_MKGains(self):
        self.assertTrue(len(catalog_reader.catalog_factory.from_MKGains(fluxlim15=0.1)) > 1)