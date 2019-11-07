# For development purposes, use the 'develop' target to install this package from the source repository (rather than the public python packages). This
# means that local changes to the code will automatically be used by the package on your machine.
# To do this type
#     make develop
# in this directory.
develop:
	sudo python3 setup.py develop

lint:
	pylint --extension-pkg-whitelist=numpy --ignored-modules=numpy,tart_tools --extension-pkg-whitelist=astropy tart2ms

test_upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository pypi dist/*

JSON='/home/tim/github/TART/doc/misc/measurement_set/data_2019_08_04_21_38_31_UTC.json'
test:
	rm -rf test.ms
	tart2ms --json ${JSON} --ms test.ms
	wsclean -name test -size 1280 1280 -scale 0.0275 -niter 0 test.ms

test2:
	rm -rf test.ms
	tart2ms --json ${JSON} --ms test.ms --pol2
	wsclean -name test -size 1280 1280 -scale 0.0275 -niter 0 test.ms
