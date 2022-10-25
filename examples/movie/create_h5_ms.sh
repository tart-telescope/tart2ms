#!/bin/bash
export OPENBLAS_NUM_THREADS=1;

FILES="/home/tim/Downloads/tart_obs/tart_obs/*.hdf"
i=0
for h in $FILES
do
  echo $h
  ms="tart_obs_hdf_$i.ms"
  tart2ms --hdf $h --ms $ms
  let i++
    wsclean -name obs_hdf_$i -weight briggs 0 -pol RR -size 1000 1000 -scale 0.175 -niter 1000 -gain 0.1 -mgain 0.05 -padding 1.5 -auto-mask 7 $ms
    convert obs_hdf_$i-image.fits obs_hdf_$i.jpeg
done

