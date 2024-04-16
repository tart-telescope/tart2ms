#!/bin/bash
# pip3 install tart_tools --upgrade
# Needs v1.1.2b1 to fix a bug when checksumming partially downloaded local files.
TART_API=https://tart.elec.ac.nz/${TARGET}/
DIR=data_${TARGET}
rm -rf ${DIR}
mkdir -p ${DIR}
VENV=~/.tartvenv/bin
cd ${DIR}
for i in {1..60}
do
  ${VENV}/tart_download_data --api ${TART_API} --n 1 --vis --file obs_$i.hdf
  ${VENV}/tart_calibration_data --n 1 --file obs_$i.json
  sleep 60
done
