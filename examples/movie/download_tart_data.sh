#!/bin/bash
# pip3 install tart_tools --upgrade
# Needs v1.1.2b1 to fix a bug when checksumming partially downloaded local files.
TART_API=https://tart.elec.ac.nz/${TARGET}/
for i in {1..30}
do
  tart_download_data --api ${TART_API} --n 1 --vis
  sleep 1
  mv `find . -name 'vis_*.hdf'` obs_$i.hdf
  tart_calibration_data --n 1 --file obs_$i.json
  sleep 120
done
