import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, EarthLocation, Angle
from casacore.tables import table


logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_array_location(ms_file):
    ms = table(ms_file)
    ant = table(ms.getkeyword("ANTENNA"))
    ant_p = ant.getcol("POSITION")
    ms.unlock()
    ms.close()
    p = np.mean(ant_p, axis=0)
    loc = EarthLocation.from_geocentric(p[0], p[1], p[2], 'm')
    geo = loc.to_geodetic(ellipsoid='WGS84')
    logger.info(f"Telescope center : {np.mean(ant_p, axis=0)}")
    logger.info(f"Telescope center : {geo}")
    return {"lon": geo.lon.to_value('deg'),
            "lat": geo.lat.to_value('deg'),
            "height": geo.height.to_value('m')
            }
