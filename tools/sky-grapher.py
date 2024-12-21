#!/usr/bin/env python
# sky-grapher: graph grid output across the celestial sky
# 02024-07-29 CE (JD 2460521)
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

    def set_vals(self, rho, f, lat, lon):
        self.rho_vals_all = rho
        self.f_vals = f
        self.lat_vals_all = lat
        self.lon_vals_all = lon
        self.unique_f = np.unique(self.f_vals)

    def print_unique(self):
        print(f'N of unique frequencies: {len(self.unique_f)}')

    def set_closest_f(self, given_index):
        print('current frequency:', given_index)
        assert isinstance(given_index, int),\
            "Given index to graph must be an integer"
        self.closest_f_idx = given_index


def graph_f_ind(results_object):
    """ Pull out a single frequency to graph """
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
    plot_path = Path('')
    plot_file = f'{closest_f:.6f}_Hz_plot.png'
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
        plt.savefig(plot_path.joinpath(plot_file))
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
        plt.savefig(plot_path.joinpath(plot_file))
        plt.close()
    else:
        print('Fewer coordinate pairs than valid indices: should not happen')

# XXX: doesn't work under the current data storage system
def test_grid(f_central = 0.006, all_vals = False):
    """ New example of graphing the whole sky, rectangular """
    filename = 'grid_output.hdf5'
    #file_path = Path(__file__).parent.parent.joinpath('examples')
    #data_array = h5py.File(Path(file_path.joinpath(filename)))
    data_array = h5py.File(filename,'r')
    # new way to do it
    data = data_array
    f_vals = data['data']['0']['toplist'][0][0]
    rho_ave_vals_all = data['data']['0']['toplist'][0][1]
    rho_circ_vals_all = data['data']['0']['toplist'][0][2]
    len_data = len(data['data'])
    fdot_vals_all = [data['data'][str(qq)]['fdot'][()] for qq in range( len_data )]
    lat_vals_all = [data['data'][str(qq)]['lat'][()] for qq in range( len_data)]
    lon_vals_all = [data['data'][str(qq)]['lon'][()] for qq in range( len_data )]
    print('all outputs', rho_ave_vals_all, fdot_vals_all, lat_vals_all)
    # to get rho ave graphed,
    rho_vals_all = rho_ave_vals_all
    

    # classic way to do it
    #rho_vals_all = data_array['toplist']['rho_ave'][:]
    #f_vals = data_array['toplist']['f'][:]
    #lat_vals_all = data_array['toplist']['lat'][:]
    #lon_vals_all = data_array['toplist']['lon'][:]
    # construct a results object
    these_results = Results()
    these_results.set_vals(rho_vals_all, f_vals, lat_vals_all, lon_vals_all)
    these_results.print_unique()
    if all_vals:
        for qq in np.arange(len(these_results.unique_f)):
            closest_f_idx = int(qq)
            this_f_central = these_results.unique_f[closest_f_idx]
            print('F-central and idex ------>', this_f_central, closest_f_idx)
            these_results.set_closest_f(closest_f_idx)
            graph_f_ind(these_results)
    else:
        closest_f_idx = int((np.abs(these_results.unique_f - f_central)).argmin())
        this_f_central = f_central
        print('F CENTRAL AND INDEX ------>', this_f_central, closest_f_idx)
        these_results.set_closest_f(closest_f_idx)
        graph_f_ind(these_results)


test_grid(f_central = 0.006)
# test_grid(all_vals = True)
#for vv in np.linspace(0.001, 0.010, 200):
#    test_grid(f_central = vv)

