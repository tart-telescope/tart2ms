'''
    Read data from a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019-2024.

    License. GPLv3.
'''

import logging
import numpy as np

from casacore.tables import table

logger = logging.getLogger(__name__)


def read_ms(ms_file, field_id=0, bl_max=9e99):

    ms = table(ms_file, ack=False)

    logger.debug(f"colnames{ms.colnames()}")
    logger.debug(f"keywordnames{ms.keywordnames()}")
    logger.debug(f"fields{ms.fieldnames()}")

    ant = table(ms.getkeyword("ANTENNA"), ack=False)
    ant_p = ant.getcol("POSITION")
    logger.debug("Antenna Positions {}".format(ant_p.shape))

    # Now use TAQL to select only good data from the correct field
    subt = ms.query(f"FIELD_ID=={field_id}",
                    sortlist="ARRAY_ID", columns="TIME, DATA, UVW, ANTENNA1, ANTENNA2, FLAG")

    fields = table(subt.getkeyword("FIELD"), ack=False)
    # field columns ['DELAY_DIR', 'PHASE_DIR', 'REFERENCE_DIR', 'CODE', 'FLAG_ROW', 'NAME', 'NUM_POLY', 'SOURCE_ID', 'TIME']
    phase_dir = fields.getcol("PHASE_DIR")[field_id][0]
    name = fields.getcol("NAME")[field_id]
    field_time = fields.getcol("TIME")[field_id]
    logger.debug(f"Field {name} (index {field_id}): Phase Dir {np.degrees(phase_dir)}, t={field_time}")

    uvw = subt.getcol("UVW")
    logger.debug(f"uvw {uvw.shape}")

    ant1 = subt.getcol("ANTENNA1")
    ant2 = subt.getcol("ANTENNA2")
    logger.debug(f"ant = {ant1.shape}")

    flags = subt.getcol("FLAG")
    logger.debug(f"flags = {flags.shape}")

    raw_vis = subt.getcol("DATA")

    try:
        # Deal with the case where WEIGHT_SPECTRUM is not present.s
        subt = ms.query(f"FIELD_ID=={field_id}",
                        sortlist="ARRAY_ID", columns="WEIGHT_SPECTRUM")
        weight_spectrum = subt.getcol("WEIGHT_SPECTRUM")[:, :, :][:, :]
    except RuntimeError as e:
        logger.debug(f"{e}")
        weight_spectrum = np.ones_like(raw_vis)

    flags = flags[:, :, :][:, :]
    uvw = uvw[:, :]
    logger.debug("uvw {}".format(uvw.shape))
    ant1 = ant1[:]
    ant2 = ant2[:]

    # Create datasets representing each row of the spw table
    spw = table(ms.getkeyword("SPECTRAL_WINDOW"), ack=False)
    logger.debug(spw.colnames())

    frequencies = spw.getcol("CHAN_FREQ")[0]

    logger.debug(f"Frequencies = {frequencies.shape}")
    logger.debug(f"NUM_CHAN = {np.array(spw.NUM_CHAN[0])}")

    logger.debug("Flags: {}".format(flags.shape))

    # Now report the recommended resolution from the data.
    # 1.0 / 2*np.sin(theta) = limit_u
    limit_uvw = np.max(np.abs(uvw), 0)

    bl = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 2] ** 2)
    good_data = np.array(
        np.where((bl < bl_max))
    ).T.reshape((-1,))

    logger.debug("Maximum UVW: {}".format(limit_uvw))
    logger.debug("Minimum UVW: {}".format(np.min(np.abs(uvw), 0)))

    for i in range(3):
        p05, p50, p95 = np.percentile(np.abs(uvw[:, i]), [5, 50, 95])
        logger.debug(
            "       U[{}]: {:5.2f} {:5.2f} {:5.2f}".format(i, p05, p50, p95)
        )


    logger.debug(f"Indices {good_data.shape}")

    #
    #   Now read the remaining data
    #
    ant1 = ant1[good_data]
    ant2 = ant2[good_data]
    weight_spectrum = weight_spectrum[good_data]

    raw_vis = raw_vis[good_data]
    logger.debug(f"Raw Vis {raw_vis.shape}")

    u_arr = uvw[good_data, 0]
    v_arr = uvw[good_data, 1]
    w_arr = uvw[good_data, 2]
    logger.debug(f"u_arr {u_arr.shape}")

    # return ant_p, ant1, ant2, u_arr, v_arr, w_arr, frequencies, raw_vis, corrected_vis, seconds, rms_arr
    return ant1, ant2, u_arr, v_arr, w_arr, raw_vis


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Read a measurement set.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ms', required=False, default=None, help="visibility file")
    parser.add_argument('--:s', nargs='+', type=int, help="Specify the list of :s.")
    parser.add_argument('--field', type=int, default=0, help="Use this FIELD_ID from the measurement set.")
    parser.add_argument('--arcmin', type=float, default=1.0, help="Resolution limit for baseline selection.")
    parser.add_argument('--title', required=False, default="disko", help="Prefix the output files.")
    parser.add_argument('--nvis', type=int, default=10000, help="Number of visibilities to use.")
    parser.add_argument('--snapshot', type=int, default=0, help="The snapshot to choose (index starting at 0).")

    source_json = None

    ARGS = parser.parse_args()

    num_vis = ARGS.nvis
    res_arcmin = ARGS.arcmin
    field_id = ARGS.field

    source_json = None

    ARGS = parser.parse_args()

    num_vis = ARGS.nvis
    res_arcmin = ARGS.arcmin
    field_id = ARGS.field
    logger.info("Getting Data from MS file: {}".format(ARGS.ms))

    ant_p, ant1, ant2, u_arr, v_arr, w_arr, frequencies, raw_vis, corrected_vis, tstamp, rms = casa_read_ms(
        ARGS.ms, num_vis, res_arcmin, _, snapshot=ARGS.snapshot, field_id=field_id
    )
    logger.info(f"Raw Vis {raw_vis}")
    logger.info(f"Frequency {frequencies.shape}")
    logger.info(f"tstamp {tstamp}")

    u_arr = get_wavelengths(u_arr, frequencies)
    v_arr = get_wavelengths(v_arr, frequencies)
    w_arr = get_wavelengths(w_arr, frequencies)

    plt.plot(tstamp, np.real(raw_vis), '.')
    plt.show()

    plt.plot(u_arr, v_arr, '.')
    plt.grid(True)
    plt.title("U-V coverage")
    plt.xlabel("u (m)")
    plt.ylabel("v (m)")
    plt.legend()
    plt.show()

    # plt.plot(u_arr, corrected_vis, '.')
    # plt.show()
    logger.info(f"Raw Mean {np.mean(np.abs(raw_vis))}")
    logger.info(f"Corrected Mean {np.mean(np.abs(corrected_vis))}")
    logger.info(f"Corrected sigma {np.std(np.abs(corrected_vis))}")

    figure, axes = plt.subplots()
    # plt.plot(np.real(raw_vis), np.imag(raw_vis), '.', label="Raw")
    plt.plot(np.real(corrected_vis), np.imag(corrected_vis), '.', label="Corrected")
    sdev = plt.Circle(np.mean(np.real(corrected_vis)),
                      np.mean(np.imag(corrected_vis)),
                      np.std(np.abs(corrected_vis)), fill=False)
    axes.set_aspect(1)
    axes.add_artist(sdev)
    plt.grid(True)
    plt.title(f"Raw and Corrected Complex Vis N={num_vis}")
    plt.legend()
    plt.show()
    # plot_antenna_positions(ant_p)
