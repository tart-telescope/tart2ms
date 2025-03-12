'''
    A convert TART JSON data into a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019-2024.

    License. GPLv3.
'''
from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='tart2ms',
    version='0.6.0b5',
    description='Convert TART observation data to Measurement Sets',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='http://github.com/tmolteno/tart2ms',
    author='Tim Molteno',
    author_email='tim@elec.ac.nz',
    license='GPLv3',
    install_requires=[
                    # distribution specific packages 3.8, 3.9
                    'dask-ms<=0.2.10; python_version<"3.10"',
                    'dask<=2023.5.0; python_version<"3.10"',
                    'python-casacore<=3.4.0; python_version<"3.10"',
                    'numpy<=1.21.5; python_version<"3.10"',
                    'pandas<=2.0.3; python_version<"3.10"',
                    'scipy<=1.10.1; python_version<"3.10"',
                    'astropy<=5.3.4; python_version<"3.10"',
                    # distribution specific packages 3.10, 3.11
                    'dask_ms<=0.2.21; python_version>="3.10" and python_version<"3.12"',
                    'dask<=2024.10.0; python_version>="3.10" and python_version<"3.12"',
                    'python-casacore<=3.6.1; python_version>="3.10" and python_version<"3.12"',
                    'numpy<2.0; python_version>="3.10" and python_version<"3.12"',
                    'pandas<=2.2.3; python_version>="3.10" and python_version<"3.12"',
                    'scipy<=1.15.1; python_version>="3.10" and python_version<"3.12"',
                    'astropy<=7.0.0; python_version>="3.10" and python_version<"3.12"',
                    'tart',
                    'tart_tools',
                    'h5py',
                    'progress',
                    'requests',
                    'scipy'],
    test_suite='nose.collector',
    tests_require=['nose'],
    extras_require={
        "predict": ['codex-africanus<=0.3.4; python_version<"3.10"',
                    'numba<=0.55.1; python_version<"3.10"',
                    'codex-africanus<=0.3.4; python_version>="3.10" and python_version<"3.11"',
                    'numba<=0.56.0; python_version>="3.10" and python_version<"3.11"',
                    'codex-africanus<=0.3.7; python_version>="3.11" and python_version<"3.12"',
                    'numba<=0.60.0; python_version>="3.11" and python_version<"3.12"',
                    ],
    },
    packages=['tart2ms'],
    include_package_data=True,
    scripts=['bin/tart2ms'],
    python_requires=">=3.8,<3.14",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Topic :: Communications :: Ham Radio",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Intended Audience :: Science/Research"])
