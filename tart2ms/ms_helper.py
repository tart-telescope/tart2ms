import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, EarthLocation, Angle
from astropy import units as u
from casacore.tables import table


logger = logging.getLogger("tart2ms")
logger.setLevel(logging.INFO)

def azel2radec(az, el, location, obstime):
    """ az, el -- in degrees
        location -- observer location on the ground 
        time -- astropy time for the observation
        returns radec in radians (fk5 frame)
    """
    dir_altaz = SkyCoord(alt=el*u.deg, az=az*u.deg, obstime=obstime,
                         frame='altaz', location=location)
    dir_j2000 = dir_altaz.transform_to('fk5')
    direction_src = [dir_j2000.ra.radian, dir_j2000.dec.radian]
    return direction_src

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
