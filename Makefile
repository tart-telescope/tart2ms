# For development purposes, use the 'develop' target to install this package from the source repository (rather than the public python packages). This means that local changes to the code will automatically be used by the package on your machine.
# To do this type
#     make develop
# in this directory.

MS=test.ms


develop:
	pip3 install -e .

test:
	python3 -m pytest

testms:
	rm -rf ${MS}
	barber --ms signal.ms --debug
