import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import re

from astropy.coordinates import SkyCoord, EarthLocation, Angle
from astropy import units as u
from casacore.tables import table

logger = logging.getLogger("tart2ms")
logger.setLevel(logging.INFO)

AFRICANUS_DFT_AVAIL = True
try:
    from africanus.rime.dask import wsclean_predict
    from africanus.coordinates.dask import radec_to_lm
except ImportError:
    logger.warning("Cannot import Africanus API. MODEL_DATA filling capabilities disabled")
    AFRICANUS_DFT_AVAIL = False

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

def get_observation_time(ms_file):
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

def predict_model(dask_data_shape, dask_data_chunking, dask_data_dtype,
                  uvw_data,
                  epoch_s, spw_chan_freqs, spw_i,
                  zenith_directions,
                  map_row_to_zendir,
                  location,
                  sources, epoch_s_sources, sources_obstime,
                  writemodelcatalog,
                  filter_elevation=45.,
                  filter_name=r"(?:^GPS.*)|(?:^QZS.*)"):
    if not AFRICANUS_DFT_AVAIL:
        raise RuntimeError("Cannot predict model visibilities. Please install codex-africanus package")
    if not sources:
        logger.critical("You have requested to predict a model for GNSS sources, however one or more of "
                        "the databases you've provided contains no GNSS source information. The MODEL_DATA "
                        "column of your database may be incomplete")
        return None
    else:
        if len(epoch_s_sources) != len(sources):
            raise RuntimeError(
                "If sources are specified then we expected epochs to be of same size as sources list")
        model_data = da.zeros(dask_data_shape, chunks=dask_data_chunking, dtype=dask_data_dtype)
        spwi_chan_freqs = da.from_array(spw_chan_freqs[spw_i])
        for dataset_i, data_epoch_i in enumerate(epoch_s):
            # predict closest matching source catalog epoch
            nn_source_epoch = np.argmin(abs(np.array(epoch_s_sources) - data_epoch_i))
            epoch_s_i = epoch_s_sources[nn_source_epoch]
            if sources[nn_source_epoch] is None:
                logger.critical("You have requested to predict a model for GNSS sources, however one or more of "
                                "the databases you've provided contains no GNSS source information. The MODEL_DATA "
                                "column of your database may be incomplete")
                continue
            sources_i = list(filter(lambda s: s.get('el', -90) >= filter_elevation and
                                              re.findall(filter_name, s.get('name', 'NULLPTR')), 
                                    sources[nn_source_epoch]))
            if not sources_i:
                logger.critical("You have requested to predict a model for GNSS sources, however one or more of "
                                "the databases you've provided contains no GNSS source information. The MODEL_DATA "
                                "column of your database may be incomplete")
                continue
            logger.info(f"Predicting model for source catalog epoch {epoch_s_i:.2f} for data epoch "
                        f"{data_epoch_i:.2f} (temporal difference: {abs(epoch_s_i - data_epoch_i):.2f} s)")
            # get J2000 RADEC
            sources_radec = np.empty((len(sources_i), 2))
            names = []
            for src_i, src in enumerate(sources_i):
                names.append(src['name'].replace(" ", "_"))
                # Convert to J2000
                direction_src = azel2radec(az=src['az'], el=src['el'], 
                                            location=location, obstime=sources_obstime[nn_source_epoch])
                sources_radec[src_i, :] = direction_src
            # get lm cosines to sources
            zenith_i = zenith_directions[dataset_i]
            lm = radec_to_lm(sources_radec, zenith_i)
            source_type = np.array(["POINT"] * len(sources_i))
            gauss_shape = np.stack(([0.] * len(sources_i), # maj
                                    [0.] * len(sources_i), # min
                                    [0.] * len(sources_i)), # BPA
                                    axis=-1)
            flux = np.ones(len(sources_i)) # arbitrary unitarian flux
            spi = np.zeros((len(sources_i), 1)) # flat spectrum
            reffreq = np.ones(len(sources_i)) * np.mean(spw_chan_freqs[spw_i])
            logspi = np.ones(len(sources_i), dtype=bool)
            sel = map_row_to_zendir == dataset_i
            vis = wsclean_predict(uvw_data[sel, :],
                                    lm,
                                    source_type,
                                    flux,
                                    spi,
                                    logspi,
                                    reffreq,
                                    gauss_shape,
                                    spwi_chan_freqs)
            model_data[sel, :, :] = vis

            if writemodelcatalog:
                fcatname = f"model_sources_{dataset_i}.txt"
                logger.info(f"Writing catalog '{fcatname}'")
                with open(fcatname, "w+") as f:
                    f.write("#format:name ra_d dec_d i spi freq0\n")
                    for si in range(len(sources_i)):
                        f.write(f"{names[si]} {np.rad2deg(sources_radec[si, 0])} "
                                f"{np.rad2deg(sources_radec[si, 1])} 1.0 0.0 {reffreq[si]}\n")

        return model_data
