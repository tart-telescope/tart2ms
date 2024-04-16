'''
    A convert TART JSON data into a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019-2022.

    License. GPLv3.
'''
from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(name='barber',
        version='0.1.0a1',
        description='Barber: Remove Fringes to Measurement Sets',
        long_description=readme,
        long_description_content_type="text/markdown",
        url='http://github.com/tmolteno/barber',
        author='Tim Molteno',
        author_email='tim@elec.ac.nz',
        license='GPLv3',
        install_requires=['dask-ms',
                        'python-casacore',
                        'astropy',
                        'numpy',
                        'h5py',
                        'progress',
                        'requests'],
        test_suite='nose.collector',
        tests_require=['nose'],
        extras_require={
            "predict": ["codex-africanus"],
        },
        packages=['barber'],
        include_package_data=True,
        scripts=['bin/barber'],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Topic :: Scientific/Engineering",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            "Intended Audience :: Science/Research"])
