# Library from cross-correlation analysis of LISA data
# 02024-07-29 CE (JD 2460521)
# Grant David Meadors, Peyton Johnson

import h5py
import numpy as np
#from waveform import Orbit, Binary, Waveform, tdi_AET
# depends on where it is run
from crosscorr.waveform import Orbit, Binary, Waveform, tdi_AET


class CrossCorr():
    def __init__(self, t, tdi, L, f_max, f_res, estimator, detectors,
                 center_fdot, span_fdot, delta_fdot, center_lat,
                 span_lat, delta_lat, center_lon, span_lon,
                 delta_lon, rho_threshold):
        """
        A class for carrying out a cross-correlation analysis
        """
        assert isinstance(L, float),\
            'Detector arm length (L) must be a float'
        assert isinstance(f_max, float),\
            'Maximum frequency (f_max) must be a float'
        assert isinstance(f_res, float),\
            'Frequency resolution (f_res) must be a float' 
        assert isinstance(estimator, str),\
            'Estimator choice must be a string'
        assert isinstance(detectors, str),\
            'Detector list must be a string, 1-6 letters (XYZAET)'
        assert isinstance(center_fdot, float),\
            'Frequency derivative center must be a float'
        assert isinstance(span_fdot, float),\
            'Frequency derivative span must be a float'
        assert isinstance(delta_fdot, float),\
            'Frequency derivative spacing must be a float'
        assert isinstance(center_lat, float),\
            'Latitude center must be a float'
        assert isinstance(span_lat, float),\
            'Latitude span must be a float'
        assert isinstance(delta_lat, float),\
            'Latitude spacing must be a float'
        assert isinstance(center_lon, float),\
            'Longitude center must be a float'
        assert isinstance(span_lon, float),\
            'Longitude span must be a float'
        assert isinstance(delta_lon, float),\
            'Longitude spacing must be a float'
        assert isinstance(rho_threshold, float),\
            'Rho threshold must be a float'
        assert (f_max < 0.5*(len(t) - 1) / (t[-1] - t[0])),\
            'Requested maximum frequency is above Nyquist'
        assert (f_res <= 1.0 / (t[-1] - t[0])),\
            '''Requested frequency resolution is coarser
               than intrinsic to the observing time'''
        self.t = t
        self.tdi = tdi
        self.L = L
        self.f_max = f_max
        self.f_res = f_res
        if estimator == 'median':
            from scipy.ndimage import median_filter
            self.estimator = median_filter
            self.estimator_median_flag = True
        elif estimator == 'mean':
            from scipy.ndimage import uniform_filter1d
            self.estimator = uniform_filter1d
            self.estimator_median_flag = False
        chan_to_num = dict(zip('XYZAET', np.arange(6)))
        self.detectors = [chan_to_num[vv] for vv in detectors]
        self.center_fdot = center_fdot
        self.span_fdot = span_fdot
        self.delta_fdot = delta_fdot
        self.center_lat = center_lat
        self.span_lat = span_lat
        self.delta_lat = delta_lat
        self.center_lon = center_lon
        self.span_lon = span_lon
        self.delta_lon = delta_lon
        self.rho_threshold = rho_threshold

        self.n_det = len(self.detectors)
        detector_matrix = np.ones((self.n_det, self.n_det))
        self.det_mat = np.triu(detector_matrix, 1).astype(bool)

    def run_analysis(self):
        """ Build a search using the cross-correlation statistic """
        dt = (self.t[-1] - self.t[0]) / (len(self.t) - 1)

        self.n_obs = int(np.ceil(2.0*self.f_max*dt*len(self.t)))
        resamp_dt = dt * len(self.t) / self.n_obs
        self.n_coh = int(np.ceil(1.0/(self.f_res*resamp_dt)))
        self.resamp_t = self.t[0] + resamp_dt * \
                        np.linspace(0, self.n_coh, self.n_coh)
        self.corr_fact = self.n_coh / self.n_obs
        self.ds_resamp_t = self.resamp_t[:self.n_obs]
        self.orbit = Orbit(self.ds_resamp_t, self.L)

        tdi = np.zeros((6, len(self.resamp_t)))
        tdi[0] = self.ds_zp(self.resamp_t, self.t, self.tdi[0])
        tdi[1] = self.ds_zp(self.resamp_t, self.t, self.tdi[1])
        tdi[2] = self.ds_zp(self.resamp_t, self.t, self.tdi[2])
        tdi = tdi_AET(tdi)[self.detectors]

        freq = np.fft.fftfreq(len(self.resamp_t), resamp_dt)
        self.f = np.fft.fftshift(freq)
        self.f = self.f[self.f >= 0.]
        tdi_ft = np.fft.rfft(tdi, norm='ortho')
        tdi_ft = tdi_ft[:, :(tdi_ft.shape[1] + len(self.t)%2 - 1)]

        df = (self.f[-1] - self.f[0]) / (len(self.f) - 1)
        safety_multiplier = 3
        bins = int(safety_multiplier * np.ceil(
                   self.orbit.doppler_scale * self.f[-1] / df))
        tdi_ft_sqA = tdi_ft.real**2 + tdi_ft.imag**2
        psd = self.corr_fact * resamp_dt * tdi_ft_sqA
        if self.estimator_median_flag:
            # the median function will do a 2D filter
            # unless individual dimensions specified
            bin_size = (1, bins)
        else:
            bin_size = bins
        self.S_h_invsqrt = 1.0/np.sqrt(self.estimator(psd, size=bin_size))

        # Build search grid
        templates = self.build_search_grid()

        # Write toplist to hdf5 file
        with h5py.File('grid_output.hdf5', 'a') as f: 
            # TODO: CAN BE MADE INTO MANY PARALLEL PROCESSES
            data = f.create_group('data')
            for ii, template in enumerate(templates):
                print(f'template {ii+1}/{templates.shape[0]}')
                output = self.evaluate_template(template)
                if not output is None:
                    grp = data.create_group(str(ii))
                    grp.create_dataset('fdot', data=template[3])
                    grp.create_dataset('lat', data=template[6])
                    grp.create_dataset('lon', data=template[7])
                    names = ['rho_ave', 'rho_circ', 'f']
                    dtype = [(n, float) for n in names]
                    output_data = output.ravel().view(dtype)
                    grp.create_dataset('toplist', data=output_data)

    def build_search_grid(self):
        """ Build arrays to search over a grid of binaries """
        # TODO: make search grid use something like HealPix?
        if self.delta_fdot == 0:
            n_fdot = 1
        else:
            n_fdot = int(np.ceil(self.span_fdot/self.delta_fdot))
        if self.delta_lat == 0:
            n_lat = 1
        else:
            n_lat = int(np.ceil(self.span_lat/self.delta_lat))
        if self.delta_lon == 0:    
            n_lon = 1
        else:
            n_lon = int(np.ceil(self.span_lon/self.delta_lon))
        n_templates = n_fdot * n_lat * n_lon
        print('n in fdot, lat, lon, total:', 
              n_fdot, n_lat, n_lon, n_templates)
        templates = np.zeros((8, n_templates))
        fdot = self.center_fdot + self.delta_fdot * \
               (np.arange(n_fdot) - np.floor(n_fdot/2))
        lat = self.center_lat + self.delta_lat * \
              (np.arange(n_lat) - np.floor(n_lat/2))
        lon = self.center_lon + self.delta_lon * \
              (np.arange(n_lon) - np.floor(n_lon/2))
        templates[3] = np.repeat(fdot, n_lat * n_lon) 
        templates[6] = np.tile(np.repeat(lat, n_lon), n_fdot)
        templates[7] = np.tile(lon, n_fdot * n_lat)
        print(templates[[3, 6, 7]].T)
        return templates.T

    def evaluate_template(self, template):
        """ Resample to a given template and evaluate rho there """
        wf = Waveform(self.ds_resamp_t, Binary(*template), self.orbit)
        t_ssb = self.ds_resamp_t + wf.resamp_t()
        tdi = np.zeros((6, self.n_coh))
        tdi[0] = self.ds_zp(t_ssb[0], self.t, self.tdi[0])
        tdi[1] = self.ds_zp(t_ssb[1], self.t, self.tdi[1])
        tdi[2] = self.ds_zp(t_ssb[2], self.t, self.tdi[2])
        tdi = tdi_AET(tdi)[self.detectors, :self.n_obs]
        # XXX: data to be sent to gpu for FFT testing ("tdi" on line above)


        ab_ts = wf.ab_ts[:, self.detectors]
        rms = np.sqrt(np.mean(ab_ts**2, axis=-1)).reshape((2, 2, 1))
        ts = np.zeros((2, self.n_det, self.n_coh))
        ts[..., :self.n_obs] = tdi * ab_ts / rms
        r_ft = np.fft.rfft(ts, norm='ortho')
        r_ft = r_ft[..., :(r_ft.shape[-1] + len(self.t) % 2 - 1)]
        ft = r_ft * self.S_h_invsqrt

        rho = np.zeros((2, len(self.f)))
        for uu in np.arange(self.n_det):
            a_u = np.conj(ft[0, uu])
            b_u = np.conj(ft[1, uu])
            for vv in np.arange(self.n_det):
                if self.det_mat[uu, vv]:
                    a_v = ft[0, vv]
                    b_v = ft[1, vv]
                    rho[0] += (a_u * a_v + b_u * b_v).real
                    rho[1] += (a_u * b_v - b_u * a_v).real
        rho = 0.1 * self.corr_fact * rho
        inds = np.argwhere(rho[0] > self.rho_threshold).flatten()
        n_inds = inds.shape[0]
        if n_inds > 0:
            freqs = self.f[inds.reshape((n_inds, 1))]
            return np.hstack((rho.T[inds], freqs))
        else:
            return None

    def ds_zp(self, new_ts, old_ts, data):
        """ Combine downsampling with zero-padding """
        ds_data = np.interp(new_ts[:self.n_obs], old_ts, data)
        if self.n_obs != self.n_coh:
            z_pad_data = np.zeros(self.n_coh)
            z_pad_data[:len(ds_data)] = ds_data
            return z_pad_data
        else:
            return ds_data
