#!/bin/bash
# pip3 install tart_tools --upgrade
# Needs v1.1.2b1 to fix a bug when checksumming partially downloaded local files.
for i in {1..30}
do
  tart_download_data --n 1 --vis
  tart_calibration_data --n 1 --file obs_$i.json
  sleep 120
done
