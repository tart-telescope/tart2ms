# For development purposes, use the 'develop' target to install this package from the source repository (rather than the public python packages). This means that local changes to the code will automatically be used by the package on your machine.
# To do this type
#     make develop
# in this directory.

WSCLEAN=export OPENBLAS_NUM_THREADS=1; wsclean -weight briggs 0 -name test -pol RR -size 1000 1000 -scale 0.175 -niter 1000 -gain 0.1 -mgain 0.05 -padding 1.5 -auto-mask 7
MS=test.ms


develop:
	pip3 install -e .

test:
	python3 -m pytest


lint:
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

clean:	${MS}
	${WSCLEAN} ${MS}

HDF='./test_data/vis_2025-11-01_09_29_31.140425.hdf'
h5:
	rm -rf ${MS}
	tart2ms --hdf ${HDF} --ms ${MS}  --rephase "obs-midpoint" --single-field
# 	wsclean -name test -pol RR -size 1280 1280 -scale 0.0275 -niter 0 ${MS}

JSON='./test_data/data_2019_08_04_21_38_31_UTC.json'
testms:
	rm -rf ${MS}
	tart2ms --json ${JSON} --ms ${MS}

disko:
	disko --ms ${MS} --healpix --fov 170deg --res 1deg --SVG --lasso --alpha 0.01

test2:
	rm -rf ${MS}
	tart2ms --json ${JSON} --ms ${MS} --pol2
