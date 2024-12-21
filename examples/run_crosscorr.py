#!/usr/bin/env python
# Run cross-correlation search
# 02024-07-29 CE (JD 2460521)
# Grant David Meadors, Peyton Johnson

import h5py
import argparse
import numpy as np
from pathlib import Path
from crosscorr import CrossCorr

def run_crosscorr(args):
    data_file = Path(args.pop('data_file'))
    if not data_file.is_file():
        raise FileNotFoundError(
            f'data file {data_file} was not found')
    output_dir = Path(args.pop('output_dir'))
    # TODO: allow choice of output filename
    # TODO: create output directory if does not exist
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f'output directory {output_dir} was not found') 

    # uncomment below to use full data set (like blinded)
    data = h5py.File(data_file, 'r')['obs']['tdi'][:, 0]

    # uncomment line below to use verification binary data
    # data = h5py.File(data_file, 'r')['sky']['vgb']['tdi'][:, 0]
    
    # uncomment line below to use sgb (?) data
    #data = h5py.File(data_file, 'r')['sky']['sgb']['tdi'][:, 0]

    data = data.view((float, len(data.dtype.names))).T
    # TODO: move config parameter writing to CrossCorr
    with h5py.File(output_dir.joinpath('grid_output.hdf5'), 'w-') as f:
        config = f.create_group('config')
        for key, val in args.items():
            config.create_dataset(key, data=val)
    CrossCorr(t=data[0], tdi=data[1:], **args).run_analysis()

# TODO: make CrossCorr take an output file path as an argument
parser = argparse.ArgumentParser(fromfile_prefix_chars = '@')
parser.add_argument('--data_file', type = str, default = None,
		    help = 'Location of dataset (hdf5)')
parser.add_argument('--output_dir', type = str, default = None,
                    help = 'Location to write output data (hdf5)')
parser.add_argument('--L', type = float, default = 2.5e9,
                    help = '''Detector arm length [m]
                           (default sets length to 2.5e9)''')
parser.add_argument('--f_max', type = float, default = 0.025,
                    help = '''Maximum search frequency [Hz]
                           (default sets maximum to 0.025)''')
parser.add_argument('--f_res', type = float, default = None,
                    help = '''Frequency resolution [Hz] 
                           (resolution should be 1/T_obs)''')
parser.add_argument('--estimator', type = str, default = 'mean',
		    help = '''S_h estimator function: mean/median
                           (defaults to mean estimator)''')
parser.add_argument('--detectors', type = str, default = 'AE',
		    help = 'Detector string: any XYZAET, no spaces')
parser.add_argument('--center_fdot', type = float, default = 0.0,
		    help = '''Frequency derivative search center
                           [s^-2] (default sets center to 0)''')
parser.add_argument('--span_fdot', type = float, default = 0.0,
		    help = '''Frequency derivative search span [s^-2]
			   (default sets span to 0, one point)''')
parser.add_argument('--delta_fdot', type = float, default = 0.0,
		    help = '''Frequency derivative search step [s^-2]
			   (default sets delta to 0, one point)''')
parser.add_argument('--center_lat', type = float, default = 0.0,
		    help = '''Ecliptic latitude search center [rad]
			   (default sets center to 0)''')
parser.add_argument('--span_lat', type = float, default = 0.0,
		    help = '''Ecliptic latitude search span [rad]
			   (default sets span to 0, one point)''')
parser.add_argument('--delta_lat', type = float, default = 0.0,
		    help = '''Ecliptic latitude search center [rad]
			   (default sets delta to 0, one point)''')
parser.add_argument('--center_lon', type = float, default = 0.0,
		    help = '''Ecliptic longitude search center [rad]
			   (default sets center to 0)''')
parser.add_argument('--span_lon', type = float, default = 0.0,
		    help = '''Ecliptic longitude search span [rad] 
                           (default sets span to 0, one point)''')
parser.add_argument('--delta_lon', type = float, default = 0.0,
		    help = '''Ecliptic search center [rad]
			   (default sets delta to 0, one point)''')
parser.add_argument('--rho_threshold', type = float, default = 10.0,
		    help = '''Rho statistic threshold for toplist
			   (default sets rho threshold to 10.0)''')
run_crosscorr(vars(parser.parse_args()))
