#!/usr/bin/env python
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

class SimpleResults():
    def __init__(self):
        self.data = None
        self.freq = None
        self.rhoave = None
        self.rhocirc = None

    def load_data(self):
        self.data = h5.File('grid_output.hdf5','r')

    def set_values(self, ii):
        # new way (view)
        # would need tranpose if formatted like LISA data set
        #view_data = self.data['data'][str(0)]['toplist'][()].view((float, 3)).T
        # our output does not need a transpose
        view_data = self.data['data'][str(ii)]['toplist'][()].view((float, 3))
        self.rhoave = view_data[:, 0]
        self.rhocirc = view_data[:, 1]
        self.freq = view_data[:, 2]
        self.ii = int(ii)
        # old way
        #self.rhoave = np.array([entry[0] for entry in self.data['data'][str(ii)]['toplist'][()] ])
        #self.rhocirc = np.array([entry[1] for entry in self.data['data'][str(ii)]['toplist'][()] ])
        #self.freq = np.array([entry[2] for entry in self.data['data'][str(ii)]['toplist'][()] ])

    def easy_spectrum(self):
        """ Plot the spectrum of the toplists, assuming linear ordering
        """

        plt.plot(self.freq, self.rhoave)
        plt.plot(self.freq, self.rhocirc)
        plt.grid()
        plt.savefig('rhoave_rhocirc' + str(self.ii) + '.pdf')
        plt.close()

    def state_extreme(self, threshold = None, sigma_max = False):
        if threshold is not None:
            assert isinstance(threshold, float),\
                'Threshold must be a float'
            a_fract = len(self.rhoave[self.rhoave > threshold])/len(self.rhoave)
            c_fract = len(self.rhocirc[self.rhocirc > threshold])/len(self.rhocirc)
            print('Fraction of ave, circ > threshold:', a_fract, c_fract)
        elif sigma_max:
            ave_max = (np.max(self.rhoave) - np.mean(self.rhoave)) / np.std(self.rhoave)
            circ_max = (np.max(self.rhocirc) - np.mean(self.rhocirc)) / np.std(self.rhocirc)
            print('rho ave, circ sigma (max)-->', ave_max, circ_max)
        else:
            print('-----------------')
            print('min, max: rho ave', np.min(self.rhoave), np.max(self.rhoave))
            print('min, max: rho circ', np.min(self.rhocirc), np.max(self.rhocirc))
            print('mean, std: rho ave', np.mean(self.rhoave), np.std(self.rhoave))
            print('mean, std: rho circ', np.mean(self.rhocirc), np.std(self.rhocirc))
            print('!---------------!')

    def make_histogram(self, print_now = True, show_ave = True, show_circ = True):
        if show_ave:
            binsave, edgeave = np.histogram(self.rhoave, bins=100, density=True)
            plt.plot(edgeave[:-1], binsave)
            if print_now:
                plt.grid()
                plt.savefig('hist_rhoave' + str(self.ii) + '.pdf')
                plt.close()
            else:
                pass
        else:
            pass
        if show_circ:
            binscirc, edgecirc = np.histogram(self.rhocirc, bins=100, density=True)
            plt.plot(edgecirc[:-1], binscirc)
            if print_now:
                plt.grid()
                plt.savefig('hist_rhocirc' + str(self.ii) + '.pdf')
                plt.close()
            else:
                pass
        else:
            pass

    def make_overlapping_histogram(self, begin, limit):
        for ii in range(begin, limit):
            self.set_values(ii)
            self.make_histogram(print_now = False, show_circ = False)
        plt.grid()
        plt.savefig('hist_rhoave_all.pdf')
        plt.close()
        for ii in range(begin, limit):
            self.set_values(ii)
            self.make_histogram(print_now = False, show_ave = False)
        plt.grid()
        plt.savefig('hist_rhocirc_all.pdf')
        plt.close()


def run_spectrum(begin = 0, limit = 25):
#def run_spectrum(begin = 0, limit = 70392):
    results = SimpleResults()
    results.load_data()
    for ii in range(begin, limit):
        results.set_values(ii)
        #results.easy_spectrum()
        #results.state_extreme()
        results.state_extreme(sigma_max = True)
        #results.state_extreme(threshold = 0.1)
        #results.make_histogram()
    #results.make_overlapping_histogram(begin, limit)

run_spectrum()
