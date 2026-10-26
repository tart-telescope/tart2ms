"""
A convert TART JSON data into a measurement set.
Author: Tim Molteno, tim@elec.ac.nz
Copyright (c) 2019-2023.

License. GPLv3.
"""

import os

# Use the casacure (Rust) table backend for dask-ms. This must happen before
# anything imports casacore.tables: importing daskms aliases casacore.tables
# to casacure, and the alias is skipped if real casacore is already loaded.
os.environ.setdefault("DASK_MS_BACKEND", "casacure")
import daskms  # noqa: E402,F401

from .casa_read_ms import read_ms as casa_read_ms
from .ms_helper import get_array_location
from .read_ms import read_ms
from .tart2ms import DEFAULT_CHUNK_SIZE, MS_STOKES_ENUMS, ms_from_hdf5, ms_from_json
