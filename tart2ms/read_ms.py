'''
    Read data from a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019-2022.

    License. GPLv3.
'''

import dask
import logging
import datetime
import time
import os

import numpy as np

from daskms import xds_from_table, xds_from_ms


from .util import resolution_min_baseline, rayleigh_criterion

logger = logging.getLogger("tart2ms")


class RadioObservation(object):
    def __init__(self):
        pass

def dt(_t0):
    return f"  \t\t\tElapsed {time.perf_counter() - _t0 :04f} s"

def read_ms(ms, num_vis, angular_resolution, channel=0, field_id=0, ddid=0, snapshot=0, pol=0):
    """
    Use dask-ms to load the necessary data to create a telescope operator
    (will use uvw positions, and antenna positions)

    -- angular_resolution: Angular resolution (degrees) Used to calculate the maximum UVW baselines to include.
    """

    # local_cluster = distributed.LocalCluster(processes=False)
    # address = local_cluster.scheduler_address
    # logging.info("Using distributed scheduler "
    # "with address '{}'".format(address))
    # client = distributed.Client()
    if not os.path.exists(ms):
        raise RuntimeError(f"Measurement set {ms} not found")
    chunks=10000
    try:
        _tic = time.perf_counter()
        logger.info(f"Reading {ms}...")

        # Create a dataset representing the entire antenna table
        ant_ds, tabkw, colkw = xds_from_table(f"{ms}::ANTENNA",
                                              table_keywords=True,
                                              column_keywords=True)
        logger.info(f"Table Keywords: {tabkw}")
        logger.info(f"Column Keywords: {colkw}")

        ant_p = np.array(ant_ds[0].POSITION.data)
        # dt = 7.116458
        logger.info(f"Antenna Positions {ant_p.shape},  {dt(_tic)}")

        # Create a dataset representing the field
        for field_ds in xds_from_table(f"{ms}::FIELD"):
            if field_id >= field_ds.sizes['row'] or field_id < 0:
                raise RuntimeError(f"Selected field {field_id} is not a valid field identifier. "
                                   f"Must be in [0, {field_ds.sizes['row']-1}]")
            phase_dir = np.array(field_ds.PHASE_DIR.data)[field_id].flatten()
            name = field_ds.NAME.data.compute()[field_id]
            logger.info("Field {} (index {}): Phase Dir {}".format(name,
                                                                   field_id,
                                                                   np.degrees(phase_dir)))

        # Create datasets representing each row of the spw table
        # we need a map to select SPW based on DDID first
        # MAIN.DDID (FK) -> DATA_DESCRIPTOR.SPECTRAL_WINDOW_ID (FK) -> SPECTRAL_WINDOW.CHAN_FREQ
        logger.info(f"Getting Data Desciptions..,  {dt(_tic)}")
        ddid_table = "::".join((ms, "DATA_DESCRIPTION"))
        for ddid_ds in xds_from_table(ddid_table, group_cols="__row__"):
            spw_ids = ddid_ds.SPECTRAL_WINDOW_ID.data.compute()
            if ddid < 0 or ddid >= ddid_ds.sizes['row']:
                raise RuntimeError(f"Selected DDID {ddid} is not a valid DDID identifier. "
                                   f"Must be in [0, {ddid_ds.sizes['row']-1}]")
            logger.info(f"Selecting Data Descriptor ID {int(ddid)} per user request")
            logger.info(f"DDID {int(ddid)} selects IF / SPW ID {int(spw_ids[ddid])}")
            # may be needed if we use mixed receivers in the future??
            # although I suspect GNSS receivers are all just circular
            pol_ids = ddid_ds.POLARIZATION_ID.data.compute()

        logger.info(f"Getting Spectral Window..,  {dt(_tic)}")
        spw_table = "::".join((ms, "SPECTRAL_WINDOW"))
        for spw_ds in xds_from_table(spw_table, group_cols="__row__"):
            logger.info("CHAN_FREQ.shape: {}".format(spw_ds.CHAN_FREQ.values.shape))
            frequencies = dask.compute(spw_ds.CHAN_FREQ.values)[int(spw_ids[ddid])].flatten()
            frequency = frequencies[channel]
            logger.info("Selected SPW {} Frequencies = {} MHz".format(spw_ids[ddid],
                                                                      ",".join(map(lambda nu: f"{nu:.3f}",
                                                                               frequencies * 1e-6))))
            logger.info("Selected imaging Frequency = {}".format(frequency))
            logger.debug("NUM_CHAN = %f" % np.array(spw_ds.NUM_CHAN.values)[0])

        # Create datasets from a partioning of the MS
        logger.info(f"Getting datasets..,  {dt(_tic)}")
        group_cols = ["FIELD_ID", "DATA_DESC_ID"]
        datasets = list(xds_from_ms(ms, chunks={"row": chunks}, group_cols=group_cols))
        logger.debug("DataSets: N={}".format(len(datasets)))

        pol = 0

        #
        # Helper to read numpy arrays. This is VERY SLOW taking
        # around 10 seconds to perform on an array of ten minutes
        # of TART visibilities. (50000 rows)
        #
        def read_np_array(da, title, dtype=np.float32):
            tic = time.perf_counter()
            logger.info("Reading {}...".format(title))
            ret = np.array(da, dtype=dtype)
            toc = time.perf_counter()
            logger.info(f"Shape {ret.shape} time {toc - tic :04f} s")
            return ret

        no_datasets_read = 0
        logger.info(f"Processing datasets..,  {dt(_tic)}")
        for ds in datasets:
            logger.debug(
                "DATASET field_id={} shape: {}".format(ds.FIELD_ID, ds.DATA.data.shape)
            )
            logger.debug("UVW shape: {}".format(ds.UVW.data.shape))
            logger.debug("SIGMA shape: {}".format(ds.SIGMA.data.shape))
            if int(field_id) == int(ds.FIELD_ID) and \
               int(ddid) == int(ds.DATA_DESC_ID):
                no_datasets_read += 1
                logger.info(f"Found DATASET field_id={ds.FIELD_ID} shape: {ds.DATA.data.shape}")
                uvw = read_np_array(ds.UVW.data, "UVW")
                flags = read_np_array(
                    ds.FLAG.data[:, channel, pol], "FLAGS", dtype=np.int32
                )

                #
                #
                #   Now calculate which indices we should use to get the required number of
                #   visibilities.
                #
                bl_max = resolution_min_baseline(max_freq=frequency,
                                                 resolution_deg=angular_resolution)

                logger.info("Resolution Max UVW: {:g} meters".format(bl_max))
                logger.info("Flags: {}".format(flags.shape))

                # Now report the recommended resolution from the data.
                # 1.0 / 2*np.sin(theta) = limit_u
                limit_uvw = np.max(np.abs(uvw), 0)

                res_limit = rayleigh_criterion(baseline_lengths=[np.max(limit_uvw)],
                                               max_freq=frequency)
                logger.info(f"Nyquist resolution: {res_limit*60 :6.3f} arcmin")

                if True:
                    bl = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 2] ** 2)
                    # good_data = np.array(np.where((flags == 0) & (np.max(np.abs(uvw), 1) < bl_max))).T.reshape((-1,))
                    good_data = np.array(
                        np.where((flags == 0) & (bl < bl_max))
                    ).T.reshape((-1,))
                else:
                    good_data = np.array(np.where(flags == 0)).T.reshape((-1,))
                logger.info("Good Data {}".format(good_data.shape))

                logger.info("Maximum UVW: {}".format(limit_uvw))
                logger.info("Minimum UVW: {}".format(np.min(np.abs(uvw), 0)))

                for i in range(3):
                    p05, p50, p95 = np.percentile(np.abs(uvw[:, i]), [5, 50, 95])
                    logger.info(
                        "       U[{}]: {:5.2f} {:5.2f} {:5.2f}".format(i, p05, p50, p95)
                    )

                n_max = len(good_data)

                if n_max <= num_vis:
                    indices = good_data
                else:
                    indices = np.random.choice(
                        good_data, min(num_vis, n_max), replace=False
                    )
                    # sort the indices to keep them in order (speeds up IO)
                    indices = np.sort(indices)
                #
                #
                #   Now read the remaining data
                #
                sigma = read_np_array(ds.SIGMA.data[indices, pol], "SIGMA")
                # ant1   = read_np_array(ds.ANTENNA1.data[indices], "ANTENNA1")
                # ant12  = read_np_array(ds.ANTENNA1.data[indices], "ANTENNA2")
                cv_vis = read_np_array(
                    ds.DATA.data[indices, channel, pol], "DATA", dtype=np.complex64
                )

                epoch_seconds = np.mean(np.array(ds.TIME.data))
                break   # TODO Check this TCAM. Stop once we have our field ID.

        logger.info(f"Processing complete,  {dt(_tic)}")
        if no_datasets_read == 0:
            raise RuntimeError("FIELD_ID ({}) or DDID ({}) contains no data".format(field_id, ddid))

        hdr = {
            "CTYPE1": ("RA---SIN", "Right ascension angle cosine"),
            "CRVAL1": np.degrees(phase_dir)[0],
            "CUNIT1": "deg     ",
            "CTYPE2": ("DEC--SIN", "Declination angle cosine "),
            "CRVAL2": np.degrees(phase_dir)[1],
            "CUNIT2": "deg     ",
            "CTYPE3": "FREQ    ",  # Central frequency
            "CRPIX3": 1.0,
            "CRVAL3": "{}".format(frequency),
            "CDELT3": 10026896.158854,
            "CUNIT3": "Hz      ",
            "EQUINOX": "2000.",
            "DATE-OBS": "{}".format(epoch_seconds),
            "BTYPE": "Intensity",
        }

        # from astropy.wcs.utils import celestial_frame_to_wcs
        # from astropy.coordinates import FK5
        # frame = FK5(equinox='J2010')
        # wcs = celestial_frame_to_wcs(frame)
        # wcs.to_header()

        u_arr = uvw[indices, 0].T
        v_arr = uvw[indices, 1].T
        w_arr = uvw[indices, 2].T

        rms_arr = sigma.T

        logger.info(f"vis {cv_vis.shape}")
        logger.info(f"Max vis {np.max(np.abs(cv_vis))}")

        # Convert from reduced Julian Date to timestamp.
        timestamp = datetime.datetime(
            1858, 11, 17, 0, 0, 0, tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(seconds=epoch_seconds)

    except Exception as e:
        logger.error("Exception {}".format(e), exc_info=True)
        logger.exception(e)

    # finally:
    # client.close()
    # local_cluster.close()

    return u_arr, v_arr, w_arr, frequency, cv_vis, hdr, timestamp, rms_arr, indices
