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
from datetime import datetime, timedelta, timezone

import dateutil.parser
from astropy import constants
from astropy.coordinates import SkyCoord
# from astropy import units as u

LOGGER = logging.getLogger("tart2ms")


def get_wavelength(frequency):
    return constants.c.value / frequency


def get_wavelengths(distance, frequency):
    '''
        Used to convert from meters to uvw coordinates
    '''
    return distance * frequency / constants.c.value


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

    LOGGER.debug("Baseline lengths:")
    LOGGER.debug(f"{baseline_lengths}")
    LOGGER.debug(f"\tMinimum: {min_baseline:.4f} m")
    LOGGER.debug(
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
        raise RuntimeError("named_phasings should contain attribute 'FRAME'")
    for req_key in ["RA", "DEC", "EQUINOX"]:
        if not all(map(lambda x: req_key in x['position'],
                   filter(lambda x: x['position']["FRAME"] != "Special Body", vals))):
            raise RuntimeError(f"Non special body named_phasings position should contain attribute {req_key}")
    if not all(map(lambda x: __try_construct_skycoord(x['position']),
                   filter(lambda x: x['position']["FRAME"] != "Special Body", vals))):
        raise RuntimeError("One or more positions in the named_phasings.json is not convertable to astropy SkyCoord")
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
        return None  # no match
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


# ---------------------------------------------------------------------------
# TART online archive queries (issue #44)
#
# Query syntax: <Name>:START:INTERVAL:END
#   Name      telescope name in the archive bucket (e.g. signal, rhodes, ...)
#   START     either a minute offset relative to now (e.g. -10 for ten minutes
#             ago) or an ISO-8601 timestamp
#   INTERVAL  sampling interval in minutes
#   END       same syntax as START (e.g. 0 for now)
#
# Timestamps may themselves contain colons (e.g. 2022-08-17T15:14:58), so the
# fields cannot simply be split on ':'. The grammar below matches each field
# shape explicitly instead.
# ---------------------------------------------------------------------------
_ISO_TIMESTAMP = (
    r"\d{4}-\d{2}-\d{2}T\d{2}"
    r"(?::\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"(?:Z|[+-]\d{2}(?::\d{2})?)?"
)
_OFFSET_MINUTES = r"-?\d+(?:\.\d+)?"
_ARCHIVE_QUERY_RE = re.compile(
    rf"^(?P<name>[^:]+)"
    rf":(?P<start>{_ISO_TIMESTAMP}|{_OFFSET_MINUTES})"
    rf":(?P<interval>{_OFFSET_MINUTES})"
    rf":(?P<end>{_ISO_TIMESTAMP}|{_OFFSET_MINUTES})$"
)


def parse_archive_query(query):
    """Parse an archive query of the form '<Name>:START:INTERVAL:END'.

    START and END are either minute offsets relative to now (e.g. -10 = ten
    minutes ago, 0 = now) or ISO-8601 timestamps. INTERVAL is the sampling
    interval in minutes. Examples:

        signal:-10:1:0          last ten minutes, sampled every minute
        rhodes:-100:10:0        last 100 minutes, sampled every 10 minutes
        signal:2022-08-17T15:14:58:5:2022-08-17T16:14:58
                                arbitrary START:INTERVAL:END window

    Returns (name, start, interval_minutes, end) with start and end left as
    raw field strings (see archive_query_window).
    """
    m = _ARCHIVE_QUERY_RE.match(query.strip())
    if m is None:
        raise ValueError(
            f"Malformed archive query '{query}'. Expected <Name>:START:INTERVAL:END "
            f"where START and END are minute offsets or ISO-8601 timestamps and "
            f"INTERVAL is the sampling interval in minutes "
            f"(e.g. 'signal:-10:1:0')"
        )
    interval = float(m["interval"])
    if interval <= 0:
        raise ValueError(
            f"Archive query '{query}' has non-positive sampling interval {interval}"
        )
    return m["name"], m["start"], interval, m["end"]


def archive_query_window(start, end, now=None):
    """Resolve the START/END fields of an archive query to UTC datetimes.

    Numeric fields are minute offsets relative to ``now`` (0 = now, -10 = ten
    minutes ago); anything else is parsed as an ISO-8601 timestamp (naive
    timestamps are assumed to be UTC).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    def resolve(field):
        if re.fullmatch(_OFFSET_MINUTES, field):
            return now + timedelta(minutes=float(field))
        ts = dateutil.parser.parse(field)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    start_utc = resolve(start)
    end_utc = resolve(end)
    if end_utc <= start_utc:
        raise ValueError(
            f"Archive query end time {end_utc.isoformat()} must be after "
            f"start time {start_utc.isoformat()}"
        )
    return start_utc, end_utc
