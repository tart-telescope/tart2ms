#!/bin/bash
export OPENBLAS_NUM_THREADS=1;

for i in {1..30}
do
  tart2ms --json ~/Downloads/tart_obs/tart_obs/obs_$i.json --ms tart_obs_$i.ms
  wsclean -name obs_$i -weight briggs 0 -pol RR -size 1000 1000 -scale 0.175 -niter 1000 -gain 0.1 -mgain 0.05 -padding 1.5 -auto-mask 7 tart_obs_$i.ms
  convert obs_$i-image.fits obs_$i.jpeg
done

