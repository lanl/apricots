# APRICOTS

## Name

APRICOTS: Averaged-Polarization Resampled Integrated Correlated Omnidirectional Test Statistics

## Description

The Averaged-Polarization Resampled Integrated Correlated Omnidirectional Test Statistics (APRICOTS) code is designed around a cross-correlation algorithm. It is designed to run on simulated data from the Laser Interferometer Space Antenna [(LISA)](https://lisa.nasa.gov/). Specifically, it has been implemented to use data from the second LISA Data Challenge (LDC), known as LDC 2 [Sangria](https://lisa-ldc.lal.in2p3.fr/challenge2a). The LDC 2 data content is fully [documented](https://lisa-ldc.lal.in2p3.fr/static/data/pdf/LDC-manual-Sangria.pdf). This simulation contains examples of most signals and noises [expected](https://doi.org/10.48550/arXiv.1702.00786) from the gravitational-wave spectrum in the LISA band.

This cross-correlation method has been focused on detecting signals from white-dwarf binary star systems, although other sources might produce a signficant signal. The white dwarf population constitutes a "galactic background" in LISA, comprised of many nearly-monochromatic "continuous wave" signals, for which cross-correlation has previously been [considered](https://doi.org/10.1103/PhysRevD.97.044017) for ground-based gravitational-wave detectors.

APRICOTS will take data in the LDC 2 format as input. The file type is expected to be HDF5. Using a configuration file, the code will run through a for-loop of "orbital" parameters, consisting of a regular, rectangular grid in frequency evolution, ecliptic latitude, and ecliptic longitude.  For each value of the orbital parameters, the LISA data channels (can be specified, defaulting to A and E) will be resampled from the LISA orbital frame to the solar-system barycenter, a process that removes the Doppler shift (Roemer delay) due to motion around the Sun and simultaneously corrects for any frequency evolution (up to the first derivative). Matched-filter time series multiply the corresponding LISA data channels, based on the plus- and cross-polarization antenna responses from that sky location. Then, the time series of the data channels are transformed into the frequency domain. The spectra are normalized by their running median in the LISA frame, which is only computed once. Finally, the spectra are cross-correlated. The cross-correlations are combined into two statistics, respectively constructed to have the best response to averaged polarization (over plus and cross) and alternately to circular polarization.

These statistics theoretically are each proportional to the square of gravitational-wave strain amplitude, but they should respond differently to the inclination angle. Information about gravitational-wave phase and the polarization angle is presumed lost. The advantage of this method is its computational efficiency over models that specify a complete waveform.

After each orbital parameter's statistics are computed, any statistics above a specified threshold are saved in an output toplist. The iteration over the orbital parameters is parallelizable, but it is currently run in sequence. Once that sequence is completed, the toplist is saved in an HDF5 output file. This file may be analyzed with the utilities in "tools" or with any applicable methods.

The fundamental tradeoff of the APRICOTS cross-correlation method is in its exploration of a multi-modal, low-dimensional parameter space, compared to the ideally-unimodal, high-dimensional parameter space of a Markov-Chain Monte Carlo method that represents all signal parameters. APRICOTS is not intended to be the most sensitive method, but it should prove extremely fast. A run over the year-long Sangria dataset, with approximately 70 thousand orbital parameters (sufficient to capture sky location to a maximum mismatch of 0.2, assuming negligible frequency evolution), requires about half a day on a few cores of an M1 Ultra CPU, utilizing only a few Gigabytes of memory.

## Installation

This code has been developed on macOS but should run cross-platform, particular on Linux and other Unix-like systems.

To install, setup an anaconda environment. Anaconda itself can be installed per the instructions on its [site](https://docs.conda.io/projects/conda).

Once this repository has been cloned, the user can then run,

```bash
conda env create -f environment.yml
```

Caution: this step may take some time. The necessary Python packages will be set up and can be activated with,

```bash
conda activate apricots
```

For the most convenient installation as a package, apricots can be set up using [Setuptools](https://setuptools.readthedocs.io/). First, modify "crosscorr/crosscorr.py" to use the "import waveform" line instead of "import crosscorr.waveform". Then run,

```bash
python setup.py install
```

Note that APRICOTS does not need to be installed as a package to run, and the instructions below assume that it is not installed, aimed at regular development.

## Usage

Examples for usage can be found for both Jupyter notebooks and command-line runs. First, download the LDC 2 Sangria dataset, as linked above. LISA Consortium login may be required for access, or the dataset may be requested (if available).

The Jupyter notebooks are found in "notebooks", containing many useful examples of graphs.

The command-line runs are more suited to an analysis of the entire dataset. From "examples", copy (cp) the files "run_crosscorr.py" and "config.ini" into the main "apricots" directory, where subsequent commands are expected to take place.

Ensure that "run_crosscorr.py" is executable:

```bash
chmod +x run_crosscorr.py
```

Modify "config.ini" so that the "--data_file" line points to the Sangria dataset. This path should include the file name and extension. Adjust any other parameters are desired, per the argument descriptions in "run_crosscorr.py" and the design of the desired analysis.

The code can then be run with,

```bash
./run_crosscorr.py @config.ini
```

Depending on input/output speed, it may take a few minutes to load the Sangria HDF5. The APRICOTS code will then launch, beginning with the line, "n in fdot, lat, lon, total: (number of each orbital parameter)". An array of the targeted orbital parameters will follow, then "template xx/(total)" as the orbital parameter templates are analyzed. On modern computing systems, each template should take only a few seconds at most, unless an extremely high frequency resolution (much finer than the default 10 nHz) is used.

At the end, a "grid_output.hdf5" file is generated. Depending on the "--rho_threshold" parameter in the configuration file, this file may contain no, some, or many entries. A full analysis of the approximately 70 thousand orbital parameters used for a rectangular sky grid with no frequency evolution at 0.2 mismatch would produce a file roughly a Terabyte in size. A rho_threshold of at least 0.05 is recommended for initial exploration.

The recommended configuration for sky location with a maximum mismatch of 0.2 is "--delta_lat=3.75e-2" and "--delta_lon=7.5e-3", with "--f_res=1.0e-8". Coarser values may be acceptable away from the ecliptic equator.

The output HDF5 file may be analyzed using the "easy_spect.py" or "quick_sky.py" scripts in "tools", or, with some modification, the "sky-grapher.py". Other tools may be written for specific analysis needs. The threshold for rho loosely corresponds to statistical significance, but it should not be directly interpreted as a probability. The stistics must be calibrated against a false alarm rate to have physical meaning.

## Support

Please contact Grant David Meadors: <gdmeadors@lanl.gov>

## Roadmap

Future contributions toward the goals of the project, particularly statistical characterization (such as a receiver operating characteristic), validation of gravitational-wave signals with simulation, and integration with the Laser Interferometer Space Antenna (LISA) Global Bayesian Fit, are encouraged. Moreover, straightforward improvements include parallelizing the for-loop over orbital parameters and making the rectangular grid into an equal-area grid, particularly for sky area. Allowing grid specification based on metric mismatch, instead of arbitrary spacing, may be useful. Computational-cost scalings can currently be estimated manually, but a calculator within the code could be calibrated to specific hardware. Any implementation on graphical processing units (GPUs) would be expected to see substantial speed improvements.

## Contributing

The project is currently is currently not accepting contributions in the main branch, but developers are welcome to fork the repository. Tasks described in the roadmap are most valuable toward the astrophysical science. If tested and validated against a simulated LISA Data Challenge set or dataset of similar complexity, these contributions may be considered.

## Authors and acknowledgment

APRICOTS has been developed by Peyton Johnson and Grant David Meadors. For more details, see the AUTHORS.md file.

## License

APRICOTS is released under the BSD-Clause License. For more details, see the LICENSE.md file.

## Project status

The project status is currently closed at the time of this commit, 2025-01-07. Developers are welcome to fork the repository for further use. Additional development may occur in the future.

This project was developed with funding from Los Alamos National Laboratory's Laboratory Directed Research & Development (LDRD) program, grant number 20230448ECR. Los Alamos National Laboratory is operated by Triad National Security, LLC, for the National Nuclear Security Administration of U.S. Department of Energy (Contract No. 89233218CNA000001).
