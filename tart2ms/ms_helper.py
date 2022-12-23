import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import re

from astropy.coordinates import (SkyCoord, 
                                 EarthLocation,
                                 Angle,
                                 AltAz,
                                 get_body)
from astropy import units as u
from casacore.tables import table
from astropy.constants import c

from tart2ms.catalogs.catalog_reader import catalog_factory
from tart2ms.fixvis import progress

logger = logging.getLogger("tart2ms")
logger.setLevel(logging.INFO)

AFRICANUS_DFT_AVAIL = True
try:
    from africanus.rime.dask import wsclean_predict
    from africanus.coordinates.dask import radec_to_lm
except ImportError:
    logger.warning("Cannot import Africanus API. MODEL_DATA filling capabilities disabled")
    AFRICANUS_DFT_AVAIL = False

def azel2radec(az, el, location, obstime, distance=None):
    """ az, el -- in degrees
        distance -- needed to convert solar system bodies back to J2000 in the barycentric frame
        location -- observer location on the ground 
        time -- astropy time for the observation
        returns radec in radians (fk5 frame)
    """
    if distance is None:
        dir_altaz = SkyCoord(alt=el*u.deg, az=az*u.deg, obstime=obstime,
                            frame='altaz', location=location)
    else:
        dir_altaz = SkyCoord(alt=el*u.deg, az=az*u.deg, obstime=obstime,
                            frame='altaz', location=location, distance=distance)
    dir_j2000 = dir_altaz.transform_to('icrs')
    direction_src = [dir_j2000.ra.radian, dir_j2000.dec.radian]
    return direction_src

def radec2azel(ra, dec, location, obstime):
    dir_j2000 = SkyCoord(ra*u.rad, dec*u.rad, frame="icrs", equinox="J2000")
    dir_altaz = dir_j2000.transform_to(AltAz(location=location, obstime=obstime[:, np.newaxis]))
    direction_src = [dir_altaz.alt.radian, dir_altaz.az.radian]
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

def get_catalog_sources_azel(timestamps, location):
    catsources = catalog_factory.from_3CRR(fluxlim15=0.5)
    catsources += catalog_factory.from_SUMMS(fluxlim15=0.5)
    catsources += catalog_factory.from_MKGains(fluxlim15=0.5)
    sources = []
    ras = list(map(lambda s: s.rarad, catsources))
    decs = list(map(lambda s: s.decrad, catsources))
    alts, azs = radec2azel(ras, decs, location=location, obstime=timestamps)
    for ti, tt in enumerate(timestamps):
        ttsources = []
        for el, az, ss in zip(alts[ti], azs[ti], catsources):
            name = ss.name
            ttsources.append({
                "name": f"CELESTIAL_{name}",
                "flux": ss.flux,
                "az": np.rad2deg(az),
                "el": np.rad2deg(el),
            })
        sources.append(ttsources)
        
    return sources

def get_solar_system_bodies(timestamps, location):
    # BH: only adding a few of the brightest right now
    # flux is estimated by RJ -- may be wildly off for the sun
    # if it flares
    class blackbody:
        radius = {
            "Moon": 1738, #km
            "Sun": 696340 #km
        }
        blackbody_temp = {
            "Moon": 270, # avg K
            "Sun": 5778 # avg K
        }
        def __init__(self, name, distance):
            self.__name = name
            self.__distance = distance
        
        def rj(self):
            """ RJ as a function of angular size (amin) and temperature (K) """
            temp = blackbody.blackbody_temp[self.__name]
            angular_size = np.rad2deg(blackbody.radius[self.__name]*2 / self.__distance) * 60. # amins
            return lambda nu: 2.65 * temp * angular_size**2 / (c.value / nu * 1e2)**2


    sources = []
    for ti, tt in enumerate(timestamps):
        ttsources = []
        for name in ["Sun","Moon"]:
            body = get_body(name, tt, location=location)
            dir_altaz = body.transform_to(AltAz(location=location, obstime=tt))
            direction_src = [dir_altaz.alt.radian, dir_altaz.az.radian]
            model = blackbody(name, body.distance.km)    
            ttsources.append({
                "name": f"SOLAR_{name}",
                "flux": model.rj(),
                "az": np.rad2deg(direction_src[1]),
                "el": np.rad2deg(direction_src[0]),
                "distance": body.distance # AU - needed for backwards conversion to J2000 epoch of non extra-galactic objects
            })            
        sources.append(ttsources)
    return sources
    
def predict_model(dask_data_shape, dask_data_chunking, dask_data_dtype,
                  uvw_data,
                  epoch_s, spw_chan_freqs, spw_i,
                  zenith_directions,
                  map_row_to_zendir,
                  location,
                  sources, epoch_s_sources, sources_obstime,
                  writemodelcatalog,
                  filter_elevation=45.,
                  filter_name=r"(?:^GPS.*)|(?:^QZS.*)|(?:^CELESTIAL_.*)|(?:^SOLAR_.*)",
                  cat_name_prefix="model_sources_",
                  append_catalog=False,
                  default_flux=1e5):
    if not AFRICANUS_DFT_AVAIL:
        raise RuntimeError("Cannot predict model visibilities. Please install codex-africanus package")
    if not sources:
        logger.critical("You have requested to predict a model for catalog sources, however one or more of "
                        "the databases you've provided contains no catalog source information. The MODEL_DATA "
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
                logger.critical("You have requested to predict a model for catalog sources, however one or more of "
                                "the databases you've provided contains no catalog source information. The MODEL_DATA "
                                "column of your database may be incomplete")
                continue
            sources_i = list(filter(lambda s: s.get('el', -90) >= filter_elevation and
                                              re.findall(filter_name, s.get('name', 'NULLPTR')), 
                                    sources[nn_source_epoch]))
            if not sources_i:
                logger.critical("You have requested to predict a model for catalog sources, however one or more of "
                                "the databases you've provided contains no catalog source information. The MODEL_DATA "
                                "column of your database may be incomplete")
                continue
            if abs(epoch_s_i - data_epoch_i) > 60.0:
                logger.info(f"Predicting model for source catalog epoch {epoch_s_i:.2f} for data epoch "
                            f"{data_epoch_i:.2f} (temporal difference: {abs(epoch_s_i - data_epoch_i):.2f} s)")
            # get J2000 RADEC
            sources_radec = np.empty((len(sources_i), 2))
            names = []
            for src_i, src in enumerate(sources_i):
                names.append(src['name'].replace(" ", "_").replace("CELESTIAL_", "").replace("SOLAR_", ""))
                # Convert to J2000
                direction_src = azel2radec(az=src['az'], el=src['el'], distance=src.get('distance', None), 
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
            flux = np.ones(len(sources_i))
            for ssi, ss in enumerate(sources_i):
                flux[ssi] = ss.get("flux", lambda nu: default_flux)(np.mean(spw_chan_freqs[spw_i]))
            
            spi = np.zeros((len(sources_i), 1)) # flat spectrum
            reffreq = np.ones(len(sources_i)) * np.mean(spw_chan_freqs[spw_i])
            logspi = np.ones(len(sources_i), dtype=bool)
            sel = map_row_to_zendir.compute() == dataset_i            
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
                fcatname = f"{cat_name_prefix}{dataset_i}.txt"
                with open(fcatname, "w+" if not append_catalog else "a") as f:
                    f.write("#format:name ra_d dec_d i spi freq0\n")
                    for si in range(len(sources_i)):
                        f.write(f"{names[si]} {np.rad2deg(sources_radec[si, 0])} "
                                f"{np.rad2deg(sources_radec[si, 1])} {flux[si]} 0.0 {reffreq[si]}\n")
        if writemodelcatalog:
            fcatname = f"{cat_name_prefix}*.txt"
            logger.info(f"Writing catalogs as '{fcatname}'")
        return model_data
