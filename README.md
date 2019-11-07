# tart2ms

Convert data from a [TART radio telescope](https://tart.elec.ac.nz) to measurement set format. This module relies on the excellent dask-ms module as a helper to create the measurement sets. 

## Usage

    usage: tart2ms [-h] [--json JSON] [--ms MS] [--api API] [--catalog CATALOG]
                [--vis VIS]

    Generate measurement set from a JSON file from the TART radio telescope.

    optional arguments:
    -h, --help         show this help message and exit
    --json JSON        Snapshot observation JSON file (visiblities, positions
                        and more). (default: None)
    --ms MS            Output MS table name. (default: tart.ms)
    --api API          Telescope API server URL. (default:
                        https://tart.elec.ac.nz/signal)
    --catalog CATALOG  Catalog API URL. (default:
                        https://tart.elec.ac.nz/catalog)
    --vis VIS          Use a local JSON file containing the visibilities for
                        visibility data (default: None)

## Credits

Thanks to Simon Perkins and Oleg Smirnov for help in interpreting the measurement set documentation.
