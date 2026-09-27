#!/usr/bin/env python
#
# Convert data from the TART telescope to measurement set format
#
import argparse
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime as dt

from tart.operation import settings
from tart_tools import api_handler, api_imaging

from tart2ms import ms_from_hdf5, ms_from_json, util
from tart2ms.catalogs.catalog_reader import catalog_factory
from tart2ms.tart2ms import DEFAULT_CHUNK_SIZE

logger = logging.getLogger("tart2ms")


def download_archive_files(queries, out_dir):
    """Download HDF5 visibility files matching the given TART online archive
    queries ('<Name>:START:INTERVAL:END', see util.parse_archive_query) into
    out_dir. Returns the downloaded file names in chronological order
    (per query).
    """
    try:
        from tart_tools.archive_handler import handle_archive_request
    except ImportError as e:
        raise RuntimeError(
            "--archive-search requires the 'minio' package to access the TART "
            "online archive (pip install minio)"
        ) from e
    files = []
    for qi, query in enumerate(queries):
        name, start_spec, interval, end_spec = util.parse_archive_query(query)
        start_utc, end_utc = util.archive_query_window(start_spec, end_spec)
        duration_min = (end_utc - start_utc).total_seconds() / 60.0
        # The archive holds roughly one visibility file per minute, so the
        # requested sampling interval translates to a request for
        # duration/interval observations (interval <= 1 min keeps everything)
        n_obs = -1 if interval <= 1.0 else max(int(round(duration_min / interval)), 1)
        logger.info(
            f"Searching TART archive '{name}' from {start_utc.isoformat()} to "
            f"{end_utc.isoformat()} (sampling interval {interval} min)"
        )
        before = set(os.listdir(out_dir))
        handle_archive_request(
            target=name,
            num_observations=n_obs,
            output_dir=out_dir,
            start_str=start_utc.isoformat(),
            duration_str=str(duration_min),
            file_prefix=f"q{qi}_{name}_",
        )
        new_files = sorted(set(os.listdir(out_dir)) - before)
        if not new_files:
            logger.warning(f"No archive files found for query '{query}'")
        files += [os.path.join(out_dir, fn) for fn in new_files]
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Generate measurement set from a JSON file from the TART radio telescope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        required=False,
        default=None,
        nargs="*",
        help="Snapshot observation JSON file (visiblities, positions and more).",
    )
    parser.add_argument(
        "--hdf",
        required=False,
        default=None,
        nargs="*",
        help="Visibility hdf5 file (One minutes worth of visibility data).",
    )
    parser.add_argument(
        "--archive-search",
        required=False,
        default=None,
        nargs="*",
        help="Query hdf5 files from the TART online archive and convert them. "
        "Each query has the form '<Name>:START:INTERVAL:END' where <Name> is the "
        "telescope name in the archive bucket (e.g. signal, rhodes, stellenbosch), "
        "START and END are either minute offsets relative to now (e.g. -10 for ten "
        "minutes ago, 0 for now) or ISO-8601 timestamps, and INTERVAL is the "
        "sampling interval in minutes. "
        "(-10:1:0 last ten minutes interval 1 min) -100:10:0 last 100 minutes "
        "interval 10 min. START:INTERVAL:END arbitrary timestamps.",
    )
    parser.add_argument(
        "--ms", required=False, default="tart.ms", help="Output MS table name."
    )
    parser.add_argument(
        "--api",
        required=False,
        default="https://api.elec.ac.nz/tart/mu-udm/",
        help="Telescope API server URL.",
    )
    parser.add_argument(
        "--catalog",
        required=False,
        default="https://tart.elec.ac.nz/catalog",
        help="Catalog API URL.",
    )
    parser.add_argument(
        "--vis",
        required=False,
        default=None,
        help="Use a local JSON file containing the visibilities  for visibility data",
    )
    parser.add_argument(
        "--pol2",
        action="store_true",
        help="Fake a second polarization. Some pipelines choke if there is only one.",
    )
    parser.add_argument(
        "--clobber", "-c", action="store_true", help="Overwrite output ms if exists"
    )

    """
    New CLI:    --rephase <target> Perform rephasing of _all_ input visibility snapshots (single integration visibility fields) to the specified phase center.
                                   This does not change the number of scans that will be output. There will be one field with name J_XXXXXX.

                --single-field     Merge all the fields into a single field with a common phase center. This means that imaging software will image just this one field by default
                                   Useful for deep imaging.

    Equivalents for the old --phase_center_policy CLI

                            instantaneous-zenith    : Default options. Each snapshot is a separate field
                            rephase-obs-midpoint    : --rephase obs-midpoint --single-field
                            no-rephase-obs-midpoint : --single-field
                            <another known position>: --rephase <known position> --single-field

    Other known positions are read from named_phasings and from catalogs e.g. 3C sources
    They may also be special bodies like Sun or Moon.
    """
    catalog_positions = catalog_factory.from_3CRR(fluxlim15=0.0)
    named_positions = util.read_known_phasings()
    specpos = list(map(lambda s: s.name, catalog_positions)) + list(
        map(lambda s: s["name"], named_positions)
    )
    parser.add_argument(
        "--rephase",
        required=False,
        default=None,
        help=f"Rephase all visibilities to a new phase center. 'obs-midpoint' rephases to the zenith at the observation "
        f"midpoint for the provided databases. Otherwise another known position can be given or a twelve digit J2000 epoch coordinate "
        f"(e.g. J193900-632400). Currently recognized special positions are: {','.join(specpos)}",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Make verbose logs and store to a log file"
    )
    parser.add_argument(
        "--single-field",
        action="store_true",
        help="Write out single phase center for all observations",
    )
    parser.add_argument(
        "--telescope_name",
        dest="override_telescope_name",
        required=False,
        default="TART",
        help="Override telescope name with a JPL recognized telescope name - needed for some CASA tasks",
    )
    parser.add_argument(
        "--uncalibrated",
        dest="uncalibrated",
        required=False,
        action="store_true",
        help="Do not apply calibration solutions (store raw data)",
    )
    parser.add_argument(
        "--add-model",
        dest="addmodel",
        required=False,
        action="store_true",
        help="DFT a model of the GNSS sources into MODEL_DATA if sources are available in the input database",
    )
    parser.add_argument(
        "--write-model-catalog",
        dest="writemodelcatalog",
        required=False,
        action="store_true",
        help="Write a catalog of GNSS sources in Tigger LSM format to the current path. Only of use in combination with --add-model",
    )
    parser.add_argument(
        "--skip-online-source-catalog",
        dest="no_fetch_sources",
        required=False,
        action="store_true",
        help="Skips fetching a catalog of GNSS sources online (not recommended for HDF5 based loading)",
    )
    parser.add_argument(
        "--skip-celestial-source-catalog",
        dest="no_celestial_sources",
        required=False,
        action="store_true",
        help="Skips predicting stronger celestial source prediction from 3CRR, etc. catalogs",
    )
    parser.add_argument(
        "--no-cache",
        dest="sources_recache",
        required=False,
        action="store_true",
        help="Ignores GNSS sources already cached from a previous run",
    )
    parser.add_argument(
        "--timerange-start-utc",
        dest="timerange_start_utc",
        required=False,
        default=None,
        help="Include only timestamps after this UTC time, e.g. 2022-03-23T21:08:12",
    )
    parser.add_argument(
        "--timerange-end-utc",
        dest="timerange_end_utc",
        required=False,
        default=None,
        help="Include only timestamps before this UTC time, e.g. 2022-03-23T23:30:12",
    )
    parser.add_argument(
        "--chunks-out",
        dest="chunks",
        required=False,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk sizes to use for MAIN table writeout",
    )
    parser.add_argument(
        "--override-ant-pos",
        required=False,
        default=None,
        dest="override_ant_pos",
        help="Overrides antenna positions with external json file, keyed on antenna_positions with list of ENU tripples. "
        "Normally will use the antenna positions stored in the provided h5 or json files.",
    )
    parser.add_argument(
        "--model-catalog-name-prefix",
        required=False,
        default="model_sources_",
        dest="model_catalog_name_prefix",
        help="Specify model catalog name prefix. Can be used to write model catalogs to another folder. Only effective in "
        "combination with --write-model-catalog",
    )

    ARGS = parser.parse_args()

    def __parsedateISO8601(x):
        return dt(*time.strptime(x, "%Y-%m-%dT%H:%M:%S")[:6])

    ARGS.timerange_start_utc = (
        __parsedateISO8601(ARGS.timerange_start_utc)
        if ARGS.timerange_start_utc is not None
        else None
    )
    ARGS.timerange_end_utc = (
        __parsedateISO8601(ARGS.timerange_end_utc)
        if ARGS.timerange_end_utc is not None
        else None
    )

    if ARGS.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger = logging.getLogger("tart2ms")
    logger.setLevel(level)

    fh = logging.FileHandler(filename=f"tart2ms.{dt.now().timestamp()}.log")
    fh.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("TART2MS parameters:")
    for k in vars(ARGS).keys():
        val = getattr(ARGS, k, "Information Unavailable")
        reprk = str(k).ljust(20, " ")
        logger.info(f"\t{reprk}: {val}")

    if ARGS.uncalibrated:
        logger.info("Raw data is being written to the database per user request")

    if ARGS.single_field or ARGS.rephase:
        # read special positions from catalogs
        # currently only 3CRR catalog contains named special positions
        if ARGS.rephase == "obs-midpoint":
            phase_center_policy = "rephase-obs-midpoint"
        elif ARGS.rephase is None:
            phase_center_policy = "no-rephase-obs-midpoint"
            logger.critical("Warning: No rephasing is being done. Use with caution.")
            logger.critical(
                "         The phase center will be selected as the observation midpoint. Ensure that the sky moves a small fraction"
            )
            logger.critical(
                "         of the instrument resolution (ie. should only be used for very short snapshot observations"
            )
        elif ARGS.rephase.upper() in map(
            lambda x: x.upper(),
            list(map(lambda s: s.name, catalog_positions))
            + list(map(lambda s: s["name"], named_positions)),
        ):
            phase_center_policy = f"rephase-{ARGS.rephase.upper()}"
        else:
            # finally try accept twelveball name
            rephase_coord = util.read_coordinate_twelveball(ARGS.rephase.upper())
            if rephase_coord is None:
                raise ValueError(f"Unknown rephase target {ARGS.rephase}")
            phase_center_policy = rephase_coord
    else:  # single-field is False
        phase_center_policy = "instantaneous-zenith"

    if os.path.isdir(ARGS.ms):
        if not ARGS.clobber:
            raise RuntimeError(
                "Measurement set '{}' exists. Please delete before continuing".format(
                    ARGS.ms
                )
            )
        else:
            if os.path.isdir(ARGS.ms):
                logger.info("Clobbering '{}' per user request".format(ARGS.ms))
                shutil.rmtree(ARGS.ms)
            else:
                raise RuntimeError(
                    "Measurement set path '{}' exists and is not a directory. Refusing to clobber!".format(
                        ARGS.ms
                    )
                )
    if ARGS.json and ARGS.hdf:
        raise RuntimeError(
            "At the moment can only concatenate from JSON or HDF files, not a combination of the two"
        )

    if ARGS.archive_search:
        if ARGS.json or ARGS.hdf:
            raise RuntimeError(
                "--archive-search cannot be combined with --json or --hdf"
            )
        archive_dir = tempfile.mkdtemp(prefix="tart2ms_archive_")
        logger.info(f"Downloading TART archive data to '{archive_dir}'")
        h5_files = download_archive_files(ARGS.archive_search, archive_dir)
        if not h5_files:
            raise RuntimeError(
                "No HDF5 files found in the TART archive for the given queries"
            )
        logger.info("Writing measurement set '{}'...".format(ARGS.ms))
        ms_from_hdf5(
            ARGS.ms,
            h5_files,
            ARGS.pol2,
            phase_center_policy,
            ARGS.override_telescope_name,
            applycal=not ARGS.uncalibrated,
            fill_model=ARGS.addmodel,
            writemodelcatalog=ARGS.writemodelcatalog,
            fetch_sources=not ARGS.no_fetch_sources,
            catalog_recache=ARGS.sources_recache,
            write_extragalactic_catalogs=not ARGS.no_celestial_sources,
            filter_end_utc=ARGS.timerange_end_utc,
            filter_start_utc=ARGS.timerange_start_utc,
            chunks_out=ARGS.chunks,
            override_ant_pos=ARGS.override_ant_pos,
            cat_name_prefix=ARGS.model_catalog_name_prefix,
        )

    elif ARGS.json:
        logger.info("Getting Data from file: {}".format(ARGS.json))
        logger.info("Writing measurement set '{}'...".format(ARGS.ms))
        ms_from_json(
            ARGS.ms,
            ARGS.json,
            ARGS.pol2,
            phase_center_policy,
            ARGS.override_telescope_name,
            applycal=not ARGS.uncalibrated,
            fill_model=ARGS.addmodel,
            writemodelcatalog=ARGS.writemodelcatalog,
            fetch_sources=not ARGS.no_fetch_sources,
            catalog_recache=ARGS.sources_recache,
            write_extragalactic_catalogs=not ARGS.no_celestial_sources,
            filter_end_utc=ARGS.timerange_end_utc,
            filter_start_utc=ARGS.timerange_start_utc,
            chunks_out=ARGS.chunks,
            override_ant_pos=ARGS.override_ant_pos,
            cat_name_prefix=ARGS.model_catalog_name_prefix,
        )

    elif ARGS.hdf:
        logger.info("Getting Data from file: {}".format(ARGS.hdf))
        # Load data from a HDF5 file
        logger.info("Writing measurement set '{}'...".format(ARGS.ms))
        ms_from_hdf5(
            ARGS.ms,
            ARGS.hdf,
            ARGS.pol2,
            phase_center_policy,
            ARGS.override_telescope_name,
            applycal=not ARGS.uncalibrated,
            fill_model=ARGS.addmodel,
            writemodelcatalog=ARGS.writemodelcatalog,
            fetch_sources=not ARGS.no_fetch_sources,
            catalog_recache=ARGS.sources_recache,
            write_extragalactic_catalogs=not ARGS.no_celestial_sources,
            filter_end_utc=ARGS.timerange_end_utc,
            filter_start_utc=ARGS.timerange_start_utc,
            chunks_out=ARGS.chunks,
            override_ant_pos=ARGS.override_ant_pos,
            cat_name_prefix=ARGS.model_catalog_name_prefix,
        )

    else:
        logger.info("Getting Data from API: {}".format(ARGS.api))
        api = api_handler.APIhandler(ARGS.api)
        info = api.get("info")
        ant_pos = api.get("imaging/antenna_positions")
        config = settings.from_api_json(info["info"], ant_pos)
        gains_json = api.get("calibration/gain")
        vis_json = api.get("imaging/vis")
        ts = api_imaging.vis_json_timestamp(vis_json)

        logger.info("Download Complete")

        cat_url = api.catalog_url(
            lon=config.get_lon(), lat=config.get_lat(), datestr=ts.isoformat()
        )
        src_json = api.get_url(cat_url)

        json_data = {
            "info": info,
            "ant_pos": ant_pos,
            "gains": gains_json,
            "data": [[vis_json, src_json]],
        }

        logger.info("Writing measurement set '{}'...".format(ARGS.ms))
        ms_from_json(
            ARGS.ms,
            None,
            ARGS.pol2,
            phase_center_policy,
            ARGS.override_telescope_name,
            json_data=json_data,
            applycal=not ARGS.uncalibrated,
            fill_model=ARGS.addmodel,
            writemodelcatalog=ARGS.writemodelcatalog,
            write_extragalactic_catalogs=not ARGS.no_celestial_sources,
            filter_end_utc=ARGS.timerange_end_utc,
            filter_start_utc=ARGS.timerange_start_utc,
            chunks_out=ARGS.chunks,
            override_ant_pos=ARGS.override_ant_pos,
            cat_name_prefix=ARGS.model_catalog_name_prefix,
        )
