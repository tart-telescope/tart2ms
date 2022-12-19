'''
    Utility functions for tart2ms
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019-2022.

    License. GPLv3.
'''

import logging

import numpy as np
import os
import json
import re
from astropy import constants
from astropy.coordinates import SkyCoord
from astropy import units as u

LOGGER = logging.getLogger("tart2ms")
LOGGER.setLevel(logging.INFO)


def rayleigh_criterion(max_freq, baseline_lengths):
    '''
        The accepted criterion for determining the diffraction limit to resolution
        developed by Lord Rayleigh in the 19th century.

        approx resolution given by first order Bessel functions
        assuming array is a flat disk of length max_baseline
    '''
    min_wl = constants.c.value / max_freq
    max_baseline = np.max(baseline_lengths)
    min_baseline = np.min(baseline_lengths)

    LOGGER.info("Baseline lengths:")
    LOGGER.info(f"\tMinimum: {min_baseline:.4f} m")
    LOGGER.info(
        f"\tMaximum: {max_baseline:.4f} m --- {max_baseline/min_wl:.4f} wavelengths")
    return np.degrees(1.220 * min_wl / max_baseline)


def resolution_min_baseline(max_freq, resolution_deg):
    '''
        Return the minimum baseline to achieve an angular resolution

        solve res_rad = 1.220 * min_wl / max_baseline to get

        max_baseline = 1.220 * min_wl / res_rad
    '''
    min_wl = constants.c.value / max_freq
    res_rad = np.radians(resolution_deg)

    return 1.220 * min_wl / res_rad

def read_known_phasings(fn=os.path.join(os.path.split(os.path.abspath(__file__))[0], 
                                        "named_phasings.json")):
    def __try_construct_skycoord(x):
        try:
            SkyCoord(f"{x['RA']} {x['DEC']}", equinox=x["EQUINOX"], frame=x["FRAME"])
        except:
            return False
        return True

    with open(fn, 'r') as f: 
        vals = json.load(f)
    if not isinstance(vals, list):
        raise RuntimeError("named_phasings.json should contain only a list of dictionaries")
    if not all(map(lambda x: hasattr(x, 'keys'), vals)):
        raise RuntimeError("named_phasings.json should contain only a list of dictionaries")
    for req_key in ['name', 'position']:
        if not all(map(lambda x: req_key in x.keys(), vals)):
            raise RuntimeError(f"named_phasings should contain attribute '{req_key}'")
    if not all(map(lambda x: "FRAME" in x['position'], vals)):
        raise RuntimeError(f"named_phasings should contain attribute 'FRAME'")
    for req_key in ["RA", "DEC", "EQUINOX"]:
        if not all(map(lambda x: req_key in x['position'], 
                   filter(lambda x: x['position']["FRAME"] != "Special Body", vals))):
            raise RuntimeError(f"Non special body named_phasings position should contain attribute {req_key}")
    if not all(map(lambda x: __try_construct_skycoord(x['position']),
                   filter(lambda x: x['position']["FRAME"] != "Special Body", vals))):
        raise RuntimeError(f"One or more positions in the named_phasings.json is not convertable to astropy SkyCoord")
    return vals

def read_coordinate_twelveball(coordstring):
    """
        Reads a standard twelve digit coordinate of the form JRARARA+/-DECDEC
        Acccepts J as J2000 or B as B1950 equinox
        yields ICRS Astropy SkyCoord if valid coord string is specified
        otherwise None
    """
    m = re.match(r'^(?P<equinox>J|B)(?P<ra>[0-9]{6})(?P<sign>[+-]{1})(?P<dec>[0-9]{6})$', 
                 coordstring)
    if m is None:
        return None # no match
    rah = m['ra'][0:2]
    ram = m['ra'][2:4]
    ras = m['ra'][4:6]
    sign = m['sign']
    decd = m['dec'][0:2]
    decm = m['dec'][2:4]
    decs = m['dec'][4:6]
    equinox = "J2000" if m['equinox'] == "J" else "B1950"
    # BH: use FK5 as Astropy ICRS implementation seemingly
    # discards equinox information and then convert to icrs
    # coordinate afterwards
    # for the purposes of TART (and most radio telescopes)
    # this does not make any difference ICRS ~= FK5 to few 10s mas level
    return SkyCoord(f"{rah}h{ram}m{ras}s {sign}{decd}d{decm}m{decs}s",
                    equinox=equinox,
                    frame="fk5").icrs