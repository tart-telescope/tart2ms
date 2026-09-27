## Docker container for tart2ms

The image installs tart2ms from this source tree with `uv sync --frozen`, so
it gets exactly the dependency versions pinned in `uv.lock`: dask-ms from the
[tmolteno/dask-ms](https://github.com/tmolteno/dask-ms) fork (git) using the
[casacure](https://pypi.org/project/casacure/) table backend. No casacore C++
libraries or python-casacore are needed. The build fails if dask-ms is not
running on casacure.

Build the image (the build context is the repository root):

    make build

Open a shell in the container, with the repository mounted at `/tart2ms`:

    make run

or convert data directly:

    docker compose run --rm tart2ms tart2ms --hdf /tart2ms/test_data/vis_2026-06-12_04_25_46.149086.hdf --ms /tart2ms/test.ms --clobber
