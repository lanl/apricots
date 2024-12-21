#!/usr/bin/env python
# sky-grapher: graph grid output across the celestial sky
# 2024-12-12 (JD 2460657)
# Grant David Meadors

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Results():
    def __init__(self):
        self.rho_vals_all = None
        self.f_vals = None
        self.lat_vals_all = None
        self.lon_vals_all = None
        self.closest_f_idx = 0
        self.delta_f = 0.0

    def set_vals(self, rho, f, lat, lon):
        self.rho_vals_all = rho
        self.f_vals = f
        self.lat_vals_all = lat
        self.lon_vals_all = lon
        self.unique_f = np.unique(self.f_vals)

    def set_delta(self, in_delta):
        self.delta_f = in_delta

    def print_unique(self):
        print(f'N of unique frequencies: {len(self.unique_f)}')

    def set_closest_f(self, given_index):
        print('current frequency:', given_index)
        assert isinstance(given_index, int),\
            "Given index to graph must be an integer"
        self.closest_f_idx = given_index


def graph_f_ind(results_object, new_format = True):
    """ Pull out a single frequency to graph """
    if new_format:
        #print('in grapher')
        #print(results_object.f_vals)
        #closest_f = results_object.unique_f[results_object.closest_f_idx]
        # in this case, the results object only has a few frequencies
        closest_f = results_object.closest_f_idx * \
            results_object.delta_f
        rho_vals = results_object.rho_vals_all
        lat_vals = results_object.lat_vals_all
        lon_vals = results_object.lon_vals_all
        print(rho_vals, lat_vals, lon_vals)
        valid_indices = len(rho_vals)
        assert valid_indices == len(lat_vals),\
            "Values of results object fields must match"
        assert valid_indices == len(lon_vals),\
            "Values of results object fields must match"
    else:
        closest_f = results_object.unique_f[results_object.closest_f_idx]
        delta_f = 1.0e-8
        valid_indices = \
            (results_object.f_vals > closest_f - 0.5 * delta_f) & \
            (results_object.f_vals < closest_f + 0.5 * delta_f)
        rho_vals = results_object.rho_vals_all[valid_indices]
        lat_vals = results_object.lat_vals_all[valid_indices]
        lon_vals = results_object.lon_vals_all[valid_indices]
    unique_lat, u_lat_ind = np.unique(lat_vals, return_inverse=True)
    unique_lon, u_lon_ind  = np.unique(lon_vals, return_inverse=True)
    len_unique_lat = len(unique_lat)
    len_unique_lon = len(unique_lon)
    print('length of unique lat, lon:', len_unique_lat, len_unique_lon)
    print('rho, lat, lon:', (rho_vals), (lat_vals), (lon_vals))
    print('length of rho, lat, lon:', len(rho_vals), len(lat_vals), len(lon_vals))
    # even match
    diff_size = np.sum(valid_indices) - len_unique_lat*len_unique_lon
    # print('diff in size, repeat of unique lat, lon')
    # print(diff_size)
    # print((np.sum(valid_indices)), len_unique_lat, len_unique_lon)
    if new_format == True:
        plot_path = None
    else:
        plot_path = Path('')
    plot_file = f'{closest_f:.6f}_Hz_plot.pdf'
    if diff_size == 0:
        # return None
        print('using all values')
        reshaped_lat = np.reshape(lat_vals, (len_unique_lat, len_unique_lon))
        reshaped_lon = np.reshape(lon_vals, (len_unique_lat, len_unique_lon))
        reshaped_rho = np.reshape(rho_vals, (len_unique_lat, len_unique_lon))
        for ii in range(10):
            print(reshaped_lat, reshaped_lon, reshaped_rho)
        # the logic is that for the x coordinate, each row is constant, but for
        # the y coordinate, each column is constant, in the meshgrid matrix
        # it may be safer to use the meshgrid
        # y, x = np.meshgrid()
        # basically, we should be able to reproduce the reshaped lat, lon, rho
        # 2D-arrays using meshgrid; what the above reshape command does is to
        # take the 1D lat, lon, rho arrays and then reshape them into 2D
        # whereas what meshgrid would do would be to take the unique lat, lon
        # and great the lat, lon 2D arrays. The rho, however, would have to be
        # inserted into a zero-size matrix the same as above, based on the
        # values of the correponding lat, lon
        z_min, z_max = reshaped_rho.min(), reshaped_rho.max()
        fig, ax = plt.subplots()
        x = reshaped_lon
        y = reshaped_lat
        z = reshaped_rho
        c = ax.pcolormesh(x, y, z, cmap='viridis', vmin=z_min, vmax=z_max, shading = 'nearest')
        ax.set_title('rho statistic vs sky position')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_xlabel('celestial/ecliptic longitude (rad)')
        ax.set_ylabel('celestial/ecliptic latitude (rad)')
        fig.colorbar(c, ax=ax, label = 'rho')
        if plot_path is not None:
            plt.savefig(plot_path.joinpath(plot_file))
        else:
            plt.savefig(plot_file)
        plt.close()
    elif diff_size < 0:
        # print('Fewer valid indices than unique coordinate pairs')
        # print('This is the expected case when the matrix is sparse')
        # return None
        min_lat = (unique_lat).min()
        max_lat = (unique_lat).max()
        min_lon = (unique_lon).min()
        max_lon = (unique_lon).max()
        diffs_lat = np.diff(lat_vals)
        diffs_lon = np.diff(lon_vals)
        # make absolutely sure not to divide by zero or a negative
        try:
            min_diff_lat = (diffs_lat[diffs_lat > 0.0]).min()
            min_diff_lon = (diffs_lon[diffs_lon > 0.0]).min()
        except ValueError:
            print('not enough sky locations')
            return None
        # this minimum is the grid spacing
        # we can use this information to create a map of zeros that will be our mesh
        # first, construct the 1D lat, lon arrays
        n_points_lat = 1 + int(np.round((max_lat - min_lat) / min_diff_lat))
        n_points_lon = 1 + int(np.round((max_lon - min_lon) / min_diff_lon))
        filled_lat_vals = np.linspace(min_lat, max_lat, num = n_points_lat)
        filled_lon_vals = np.linspace(min_lon, max_lon, num = n_points_lon)
        xx, yy = np.meshgrid(filled_lon_vals, filled_lat_vals)
        reshaped_lon = xx
        reshaped_lat = yy
        # have validated that the order of xx, yy increasing is as it should be
        assert len(reshaped_lon) == len(reshaped_lat),\
            'WARNING: error in creating meshgrid'
        reshaped_rho = np.zeros_like(reshaped_lat)
        # then, we have to insert the corresponding rho values
        # to do that, we first need to map the given rho's ii to
        # some point in the new mesh grid
        # look at which point in the linspace arrays are closest to originals
        # Knowing that unique lat[u lat ind] would recreate the original
        # array of latitudes, we can now do a 2-step mapping from
        # (backwards) the closest index in the filled array, to the
        # unique array, back to the original array
        lat_lookup = np.argmin(np.abs(np.subtract.outer(unique_lat, filled_lat_vals ) ), axis=1 )
        lat_lookup_array = (np.array(lat_lookup)[u_lat_ind])
        lon_lookup = np.argmin(np.abs(np.subtract.outer(unique_lon, filled_lon_vals ) ), axis=1 )
        lon_lookup_array = np.array(lon_lookup)[u_lon_ind]
        # ok, I think that this above is an array that tells each index
        # from the original indexing scheme where to go to in the
        # lat coordinate of the meshgrid
        # then, insert into the reshaped rho array of zeroes
        reshaped_rho[lat_lookup_array, lon_lookup_array] = rho_vals
        # supply to plot command
        z_min, z_max = reshaped_rho.min(), reshaped_rho.max()
        fig, ax = plt.subplots()
        x = reshaped_lon
        y = reshaped_lat
        z = reshaped_rho
        c = ax.pcolormesh(x, y, z, cmap='viridis', vmin=z_min, vmax=z_max, shading = 'nearest')
        ax.set_title('rho statistic vs sky position')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_xlabel('celestial/ecliptic longitude (rad)')
        ax.set_ylabel('celestial/ecliptic latitude (rad)')
        fig.colorbar(c, ax=ax, label = 'rho')
        if plot_path is not None:
            plt.savefig(plot_path.joinpath(plot_file))
        else:
            plt.savefig(plot_file)
        plt.close()
    else:
        print('Fewer coordinate pairs than valid indices: should not happen')



def alt_test_grid(f_central = 0.006, all_vals = False):
    """ New example of graphing the whole sky, rectangular """
    if all_vals:
        print('All values: True -> selecting first layer only')
    filename = 'grid_output.hdf5'
    #file_path = Path(__file__).parent.parent.joinpath('examples')
    data_array = h5py.File(filename,'r')
    # new way to do it
    #data = data_array
    this_data = data_array['data']
    that_data = data_array['config']
    # reveals the number of binary templates
    len_data = len(this_data)

    # pull in data
    f_max = that_data['f_max'][()]
    f_res = that_data['f_res'][()]
    span_lat = that_data['span_lat'][()] 
    span_lon = that_data['span_lon'][()]
    delta_lat = that_data['delta_lat'][()]
    delta_lon = that_data['delta_lon'][()]
    center_lat = that_data['center_lat'][()]
    center_lon = that_data['center_lon'][()]

    n_freq = int(f_max / f_res)
    #print(type(n_freq), n_freq)
    if delta_lat == 0:
        n_lat = 1 
    else:
        n_lat = int(np.ceil(span_lat/delta_lat))
    if delta_lon == 0: 
        n_lon = 1
    else:
        n_lon = int(np.ceil(span_lon/delta_lon))
    lat = center_lat + delta_lat * \
          (np.arange(n_lat) - np.floor(n_lat/2))
    lon = center_lon + delta_lon * \
          (np.arange(n_lon) - np.floor(n_lon/2))

    # but we need to store what template it is in as the first index
    # we will not accomodate multiple layers in this code
    # put lat, lon in last two fields
    ave_circ_f_array = np.zeros((len_data, 6))
    # set to blank values for size
    rho_vals_all = ave_circ_f_array[:, 0]
    f_vals = np.linspace(0.0, f_max, num = n_freq, endpoint = False)
    lat_vals_all = lat
    lon_vals_all = lon

    #print(f_vals, n_freq)
    those_results = Results()
    # will set results twice twice, even though redundant rihgt now
    those_results.set_vals(rho_vals_all, f_vals, lat_vals_all, lon_vals_all)
    those_results.print_unique()
    #print('^^^^^')
    if all_vals:
        closest_f_idx = 0
        this_f_central = those_results.unique_f[closest_f_idx]
        print('F-central and idex ------>', this_f_central, closest_f_idx)
    else:
        closest_f_idx = int((np.abs(those_results.unique_f - f_central)).argmin())
        this_f_central = f_central
        print('F CENTRAL AND INDEX ------>', this_f_central, closest_f_idx)
    those_results.set_closest_f(closest_f_idx)
    #len_unique_f = len(those_results.unique_f)
    #storage_template_arrays = np.zeros(len_unique_f)

    for qq in range(len_data):
        #print('begin template', qq)
        this_template_data = (this_data[str(qq)]['toplist'][()])
        len_qq = len(this_template_data)

        # new way, with views, faster
        this_template_view = this_template_data.view((float, 3))

        if all_vals:
            this_template_array = this_template_view[those_results.closest_f_idx]
        else:
            closest_here_idx = \
                int( (np.abs(this_template_view[:, 2] - f_central) ).argmin() )
            # return zeroes if too far away, e.g., more than a Doppler at max,
            # 1e-4 * 25 mHz = 2.5 microHz
            # alternately, be stricter, and make one bin: 1e-8
            #if ( np.abs( this_template_view[closest_here_idx, 2] - f_central ) > 2.5e-6 ) :
            if ( np.abs( this_template_view[closest_here_idx, 2] - f_central ) > 1.0e-8 ) :
                this_template_array = \
                    np.zeros(shape = np.shape(this_template_view[0]))
            else:
                # then in this case we chose the closest template
                this_template_array = this_template_view[closest_here_idx]
            #print('index', closest_here_idx)
        # old way, with lists, slower
        #this_template_array = np.array( [list(this_template_data[those_results.closest_f_idx]) ] )
        # qq reveals the binary template
        ave_circ_f_array[qq, 0] = qq
        ave_circ_f_array[qq, 1:4] = this_template_array
        ave_circ_f_array[qq, 4] = this_data[str(qq)]['lat'][()]
        ave_circ_f_array[qq, 5] = this_data[str(qq)]['lon'][()]
        
        #print('end template', qq)
    print('--> extracted')
    these_results = Results()
    # now we need the results from the specific toplists
    ave_rho_array = ave_circ_f_array[:, 1]
    f_array = ave_circ_f_array[:, 3]
    ave_lat_array = ave_circ_f_array[:, 4]
    ave_lon_array = ave_circ_f_array[:, 5]
    these_results.set_vals(ave_rho_array, f_array, ave_lat_array, ave_lon_array)
    # purely for naming the graph
    these_results.set_closest_f(closest_f_idx)
    these_results.set_delta(those_results.unique_f[1] - those_results.unique_f[0])
    #print('before grapher')
    #print(those_results.f_vals)
    graph_f_ind(these_results)


#test_grid(f_central = 0.006)
#test_grid(all_vals = True)
alt_test_grid(all_vals = True)
#alt_test_grid(f_central = 0.00622)
#for vv in np.linspace(0.001, 0.010, 200):
#    test_grid(f_central = vv)

