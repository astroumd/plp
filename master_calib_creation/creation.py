import json
import os
from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd
from astropy.modeling.polynomial import Chebyshev2D, PolynomialBase
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.optimize.optimize import OptimizeWarning
from master_calib_creation.image import ExistingImage, file_overlay, ArrayImage

from igrins.procedures.find_peak import find_peaks
from igrins.procedures.process_derive_wvlsol import fit_wvlsol
from igrins.instrument.arc import combine_lines_dat, load_lines_dat


def gen_oned_spec(
    order_map_file, twod_spectrum_file, output_file, aggregation_axis=0, aggregation=np.nanmedian
):
    """
    Creates a one dimensional spectrum for each order, and saves in json format to the output_file location.

    Parameters
    ----------
    order_map_file : str, path
        Path to fits file containing order map
    twod_spectrum_file : str, path
        Path to fits file containing 2D spectrum, should be aligned with the order map
    output_file : str, path
        Path to output json file
    aggregation_axis : integer, 0 or 1, optional
        numpy array axis to aggregate along
    aggregation : function, optional
        Aggregation function to convert 2D to 1D. np.nanmedian, np.nanmode, np.nanmean, np.nanmax, etc

    Returns
    -------

    """
    order_map_image = ExistingImage(order_map_file, fits_image_hdu=0).image
    spectrum_image = ExistingImage(twod_spectrum_file, fits_image_hdu=0).image
    gen_oned_spec_image(
        order_map_image, spectrum_image, output_file, aggregation_axis=aggregation_axis, aggregation=aggregation
    )


def gen_oned_spec_image(order_map_image, spectrum_image, output_file, aggregation_axis=0, aggregation=np.nanmedian):
    """
    Creates a one dimensional spectrum for each order, and saves in json format to the output_file location.

    Parameters
    ----------
    order_map_image : np.array
        Image containing order map
    spectrum_image : np.array
        Image containing 2D spectrum that needs reduced, should be aligned with the order map
    output_file : str, path
        Path to output json file
    aggregation_axis : integer, 0 or 1, optional
        numpy array axis to aggregate along
    aggregation : function, optional
        Aggregation function to convert 2D to 1D. np.nanmedian, np.nanmode, np.nanmean, np.nanmax, etc

    Returns
    -------

    """
    orders = get_orders_from_map(order_map_image)
    json_dict = {'specs': [], 'orders': []}
    for order in orders:
        single_order = single_order_image_from_map(spectrum_image, order, order_map_image)
        oned_spec = aggregation(single_order, axis=aggregation_axis)
        oned_spec_no_nan = fill_in_nan(oned_spec)
        json_dict['specs'].append(oned_spec_no_nan.astype(float).tolist())
        json_dict['orders'].append(int(order))
    save_dict_to_json(json_dict, output_file)


def load_polyfit(pickle_file):
    with open(pickle_file, 'rb') as f:
        polyfit = pickle.load(f)

    return polyfit


def gaussian(x, amplitude, mean, stddev, y0):
    return amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2) + y0


def gaussian2(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, y0):
    return gaussian(x, amplitude1, mean1, stddev1, y0/2) + gaussian(x, amplitude2, mean2, stddev2, y0/2)


def gaussian3(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, amplitude3, mean3, stddev3, y0):
    return gaussian(x, amplitude1, mean1, stddev1, y0/3) + gaussian(x, amplitude2, mean2, stddev2, y0/3) + \
           gaussian(x, amplitude3, mean3, stddev3, y0/3)


def curve_fit_peaks(peak, expected_width, pixels, spec, n_gauss=1, plt_peak=False, polyfit=None, order=1, plt_wvl=False, plt_pix=False, custom=tuple()):
    fit_dict = {
        1: gaussian,
        2: gaussian2,
        3: gaussian3
    }
    mean = peak[0]
    # mean_init = mean
    stddev = peak[1]
    # stddev_init = stddev
    amplitude = peak[2]
    pix = pixels[int(mean) - int(expected_width):int(mean) + int(expected_width)]
    _spec = spec[int(mean) - int(expected_width):int(mean) + int(expected_width)]
    p0_dict = {
        1: (amplitude, mean, stddev, 50),
        2: (amplitude, mean-4, stddev, amplitude, mean+4, stddev, 50),
        3: (amplitude, mean-4, stddev, amplitude, mean+4, stddev, amplitude, mean, stddev, 50),
    }
    if len(custom) > 0:
        p0 = custom
    else:
        p0 = p0_dict[n_gauss]
    popt, _ = curve_fit(fit_dict[n_gauss], pix, _spec, p0=p0)

    # amplitude, mean, stddev, x0 = popt
    # temp_filter = mean>2200 and order>25
    # if plt_peak and temp_filter:
    if plt_peak:
        print(popt)
        print(order)
        gauss_pix = np.arange(pix.min(), pix.max(), 0.1)
        if plt_pix:
            plt.scatter(gauss_pix, fit_dict[n_gauss](gauss_pix, *popt), label='fit')
            plt.scatter(pix, _spec, label='data')
            # plt.title('p_init={} p_opt={}'.format((amplitude, mean, stddev, 50), popt))
            plt.legend()
            plt.xlabel('Position, x (pixels)')
            plt.ylabel('Intensity, I (arbitrary units)')
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            plt.show()
            plt.clf()
            input("Press Enter to continue...")
        if polyfit is not None and plt_wvl:
            print(order, popt[1])
            wvls = polyfit(pix, np.ones(pix.shape)*order)/order
            gauss_wvls = polyfit(gauss_pix, np.ones(gauss_pix.shape)*order)/order
            plt.scatter(gauss_wvls, fit_dict[n_gauss](gauss_pix, *popt), label='fit')
            plt.scatter(wvls, _spec, label='data')
            plt.legend()
            plt.xlabel('Wavelength (microns)')
            plt.ylabel('Intensity')
            plt.xlim((wvls.min()-0.0001, wvls.max()+0.0001))
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            plt.show()
            plt.clf()
            input("Press Enter to continue...")
    return popt


def line_lookup(detected_peaks, line_wavelengths, line_wvl_widths, manual_filter_peaks=False, line_descriptions=None):
    """
    Finds the nearest line from the lines_dat_file for a detected peak

    Parameters
    ----------
    detected_peaks : np.array
        detected peaks from find_peaks function
    line_wavelengths : np.array
    line_wvl_widths: np.array
    manual_filter_peaks : bool
    line_descriptions: list

    Returns
    -------

    """
    if line_descriptions is None:
        line_descriptions = ['{}'.format(i) for i in range(len(line_wavelengths))]
    line_waves = []
    line_index = []
    for peak in detected_peaks:
        value, index = find_nearest(line_wavelengths, peak)
        line_waves.append(value)
        line_index.append(index)
    print('peak', 'wave', 'index', 'peak-wave')
    for peak, wave, index in zip(detected_peaks, line_waves, line_index):
        print(peak, wave, index, peak-wave)
    line_index = np.asarray(line_index)
    if manual_filter_peaks:
        cont = True
        plt.clf()
        while cont:
            for i, peak in enumerate(zip(detected_peaks, line_wvl_widths, line_index)):
                plt.errorbar(
                    [peak[0], peak[0]], [0, 1],
                    xerr=np.asarray([peak[1], peak[1]]),
                    label='peak {}, i {}'.format(i, peak[-1])
                )
            wave_indices = np.arange(line_index.min()-2, line_index.max()+3)
            waves = line_wavelengths[wave_indices]
            descriptions = np.asarray(line_descriptions)[wave_indices]
            for index, wave, description in zip(wave_indices, waves, descriptions):
                plt.plot([wave, wave], [0, 1.1], '--', label=description)
            # for i, peak in enumerate(zip(detected_peaks, line_wvl_widths, line_index)):
            #     _wvl = line_wavelengths[peak[-1]]
            #     _desc = descriptions[peak[-1]]
            #     plt.plot([_wvl, _wvl], [0, 1], ':', label=_desc)
            plt.xlim((np.min(detected_peaks)-0.003), np.max(detected_peaks)+0.003)
            plt.legend()
            plt.show()
            change = input('Current indices {}. New indices? (blank if keeping)'.format(line_index))
            if change.strip():
                split_change = change.split()
                try:
                    line_index = np.asarray([int(i) for i in split_change])
                    line_waves = line_wavelengths[line_index]
                except ValueError:
                    print("ValueError for input: {}".format(change))
            else:
                cont = False
    return np.asarray(line_waves), line_index


def parse_popt(popt):
    y0 = popt[-1]
    _amps = np.asarray(popt[:-1:3])
    _means = np.asarray(popt[1:-1:3])
    _stddevs = np.asarray(popt[2:-1:3])
    _y0s = np.asarray([y0 for i in _amps])
    return _amps, _means, _stddevs, _y0s


def calculate_wvls(pixel_positons, standard_devs, wavelengths, order, polyfit=None):
    pixpeak_start_array = pixel_positons - standard_devs
    pixpeak_end_array = pixel_positons + standard_devs
    if polyfit is None:
        detected_wvl_array = np.interp(pixel_positons, np.arange(wavelengths.shape[0]), wavelengths)
        wvlpeak_start_array = np.interp(pixpeak_start_array, np.arange(wavelengths.shape[0]), wavelengths)
        wvlpeak_end_array = np.interp(pixpeak_end_array, np.arange(wavelengths.shape[0]), wavelengths)
    else:
        detected_wvl_array = polyfit(pixel_positons, np.ones(pixel_positons.shape[0]) * order) / order
        wvlpeak_start_array = polyfit(pixpeak_start_array, np.ones(pixel_positons.shape[0]) * order) / order
        wvlpeak_end_array = polyfit(pixpeak_end_array, np.ones(pixel_positons.shape[0]) * order) / order
    wvlwidths_array = np.abs(wvlpeak_end_array - wvlpeak_start_array)
    return detected_wvl_array, wvlwidths_array


def curve_fit_peak(
        spec, peaks,  wavelengths, line_wavelengths, expected_width=12, plt_peak=False, manual_filter_peak=False,
        polyfit=None, order=1, plt_pix=False, plt_wvl=False, line_descriptions=None
):
    import warnings
    warnings.filterwarnings("error")
    if line_descriptions is None:
        line_descriptions = ['{}'.format(i) for i in range(len(line_wavelengths))]
    pixels = np.arange(len(spec))
    spec = np.asarray(spec)
    amplitudes = []
    means = []
    stddevs = []
    y0s = []
    indices = []
    waves = []
    wvl_widths = []
    detected_wvls = []
    ref_indices_grouping = []
    for peak in peaks:
        mean_init = peak[0]
        stddev_init = peak[1]
        n_gauss=1
        print(peak)
        custom = tuple()
        cont = True
        try:
            while cont:
                popt = curve_fit_peaks(
                    peak, expected_width, pixels, spec, n_gauss=n_gauss, plt_peak=plt_peak, polyfit=polyfit,
                    order=order, plt_pix=plt_pix, plt_wvl=plt_wvl, custom=custom
                )
                _amps, _means, _stddevs, _y0s = parse_popt(popt)
                detected_wvl_array, wvlwidths_array = calculate_wvls(_means, _stddevs, wavelengths, order, polyfit)

                wvl_array, ref_indices_array = line_lookup(
                    detected_wvl_array, line_wavelengths, wvlwidths_array, manual_filter_peak, line_descriptions)

                if manual_filter_peak:
                    keep_peak = input('keep peak (y,n,1,2,3,c,zi,zo)?')
                    keep_peak = keep_peak.upper().strip()
                    if keep_peak.startswith('Y') or not keep_peak:
                        keep_ref_indices = input("keep ref indices {} y/n".format(ref_indices_array))
                        if keep_ref_indices.upper().startswith('Y') or not keep_peak.strip():
                            ref_indices_grouping.append(ref_indices_array.tolist())
                        print(
                            _amps.shape, _means.shape, _stddevs.shape, _y0s.shape, wvlwidths_array.shape,
                            wvlwidths_array.shape, wvl_array.shape, ref_indices_array.shape, detected_wvl_array.shape
                        )
                        assert _amps.shape == _means.shape == _stddevs.shape == _y0s.shape == wvlwidths_array.shape\
                               == wvlwidths_array.shape == wvl_array.shape == ref_indices_array.shape\
                               == detected_wvl_array.shape
                        for _amp, _mean, _stddev, _y0, _wvl_width, _wvl, _ref_index, _detected_wvl in zip(
                                _amps, _means, _stddevs, _y0s, wvlwidths_array, wvl_array, ref_indices_array,
                                detected_wvl_array
                        ):
                            # TODO: add this back in with ability to choose a new index
                            # if _ref_index in indices:
                            #     raise Exception("Duplicate index")
                            amplitudes.append(_amp)
                            means.append(_mean)
                            stddevs.append(_stddev)
                            y0s.append(_y0)
                            wvl_widths.append(_wvl_width)
                            waves.append(_wvl)
                            indices.append(_ref_index)
                            detected_wvls.append(_detected_wvl)
                            cont = False
                    elif keep_peak.startswith('N'):
                        print('deleting peak')
                        cont = False
                    else:
                        try:
                            n_gauss = int(keep_peak.strip())
                            print('n_gauss: {}'.format(n_gauss))
                        except ValueError:
                            if keep_peak.startswith('ZI'):
                                expected_width *= 0.8
                            elif keep_peak.startswith('ZO'):
                                expected_width *= 1.5
                            else:
                                custom_string = input(
                                    'enter p0 in format "amplitude1,mean1,stddev1 amplitude2,mean2,stddev2 y0"'
                                )
                                custom_split = custom_string.split()
                                n_gauss = len(custom_split) - 1
                                custom = []
                                for i in range(n_gauss):
                                    inputs = custom_split[i].split(',')
                                    for j in inputs:
                                        try:
                                            custom.append(float(j))
                                        except ValueError:
                                            continue
                                custom.append(float(custom_split[-1]))
                else:
                    if np.abs(mean_init - _mean) > 4:  # TODO: fix the auto filter option
                        continue
                    elif np.abs(_stddev - stddev_init) > 2.0:
                        continue
        except (RuntimeError, OptimizeWarning, RuntimeWarning) as e:
            print(e)
            continue
    warnings.filterwarnings("ignore")
    line_data_array = np.asarray(
        (means, stddevs, amplitudes, y0s, waves, indices, wvl_widths, detected_wvls)).transpose()
    # TODO: remove the following np.unique debugging statements
    try:
        line_data_array_unique = np.unique(line_data_array, axis=0)
    except ValueError:
        line_data_array_unique = line_data_array
    if line_data_array.shape != line_data_array_unique.shape:
        print("Duplicates erased!")
        print(line_data_array.shape, line_data_array_unique.shape)
        print("line data array")
        for line in line_data_array:
            print(line)
        print("unique array")
        print(line_data_array_unique)
    return line_data_array_unique, ref_indices_grouping


def load_fitdata_df(
        identified_lines_json_files, ref_indices_json_files, band, centroid_solutions_json_file=None,
        domain_starting_index=4, domain_starting_pixel=0, domain_ending_pixel=None, pixels_in_order=4096
):
    identified_lines_sets = [json_dict_from_file(identified_lines_json_file) for identified_lines_json_file in
                             identified_lines_json_files]
    ref_indices_dicts = [json_dict_from_file(ref_indices_json_file)[band] for ref_indices_json_file in
                         ref_indices_json_files]
    num_orders = len(identified_lines_sets[0]['orders'])
    indices = range(num_orders)
    if centroid_solutions_json_file is not None:
        domains = json_dict_from_file(centroid_solutions_json_file)['domain']
        domains = domains[domain_starting_index:domain_starting_index + num_orders]
    else:
        if domain_ending_pixel is None:
            domain_ending_pixel = pixels_in_order
        domains = [(domain_starting_pixel, domain_ending_pixel) for j in indices]

    fitdata = {'pixels': [], 'order': [], 'wavelength': [], 'ref_indices': [], 'ref_name': []}
    for identified_lines, ref_indices_dict in zip(identified_lines_sets, ref_indices_dicts):
        for j in indices:
            order = identified_lines['orders'][j]
            domain = domains[j]
            wvls = identified_lines['wvl_list'][j]
            pixpos = identified_lines['pixpos_list'][j]
            ref_indices = identified_lines['ref_indices_list'][j]
            fit_ref_indices = ref_indices_dict.get(str(order), [])
            fit_ref_indices = [item for sublist in fit_ref_indices for item in sublist]  # flattening lists
            wvls_array = np.asarray(wvls)
            pixpos_array = np.asarray(pixpos)
            ref_indices_array = np.asarray(ref_indices)

            if len(fit_ref_indices) == 0:
                fit_indices_array = np.zeros(ref_indices_array.shape, dtype=np.bool)
            else:
                repeat_list = np.asarray([ref_indices_array == i for i in fit_ref_indices])
                fit_indices_array = np.all(repeat_list, axis=0)

            domain_indices = np.logical_and(pixpos_array > domain[0], pixpos_array < domain[1], fit_indices_array)
            pixpos = pixpos_array[domain_indices].tolist()
            wvls = wvls_array[domain_indices].tolist()
            refs = ref_indices_array[domain_indices].tolist()
            order_list = [order for i in range(len(wvls))]
            fitdata['wavelength'] = fitdata['wavelength'] + wvls
            fitdata['pixels'] = fitdata['pixels'] + pixpos
            fitdata['order'] = fitdata['order'] + order_list
            fitdata['ref_indices'] = fitdata['ref_indices'] + refs
            fitdata['ref_name'] = fitdata['ref_name'] + [identified_lines['ref_name'] for i in pixpos]

    fitdata_df = pd.DataFrame(fitdata)
    return fitdata_df


def gen_identified_lines(
        oned_spec_json_file, oned_wavemap_file, lines_dat_file, output_file, ref_indices_output, band,
        p_init_pickle=None,
        plt_peak=False, plt_wvl=False, plt_pix=False, manual_filter_peak=False, domain_starting_pixel=0,
        domain_ending_pixel=None, sigma_filter=False, peak_finder_sigma=6
):
    """
    Creates identified lines calibration file

    Parameters
    ----------
    oned_spec_json_file : str, path
        1D spectrum file containing peaks to be identified, typically generated by gen_oned_spec
    oned_wavemap_file : str, path
        1D spectrum file containing wavelength corresponding to pixel location, typically generated by gen_oned_spec
        from 2D wavemap file
    lines_dat_file : str, path
        lines list file corresponding to oned_spec_json_file spectrum
    output_file : str, path
        Path to output json file
    ref_indices_output: str, path
    band: str
    p_init_pickle: str, path
        Path to echellogram pickle file
    plt_peak :
    plt_wvl :
    plt_pix :
    manual_filter_peak :
    domain_starting_pixel :
    domain_ending_pixel :
    sigma_filter :
    peak_finder_sigma :

    Returns
    -------

    """
    if p_init_pickle is not None:
        with open(p_init_pickle, 'rb') as f:
            polyfit = pickle.load(f)
    else:
        polyfit = None
    specs, orders = parse_oned_spec(oned_spec_json_file)
    if domain_ending_pixel is None:
        domain_ending_pixel = len(specs[0])
    map_wavelengths, wavemap_orders = parse_oned_spec(oned_wavemap_file)
    order_indices = index_matching(orders, wavemap_orders)
    map_wavelengths_array = np.asarray(map_wavelengths)[order_indices]
    line_wavelengths, line_descriptions, line_intensities = load_lines_dat(lines_dat_file)
    json_dict = {
        'wvl_list': [], 'ref_name': os.path.basename(lines_dat_file), 'ref_indices_list': [], 'pixpos_list': [],
        'orders': [], 'pix_widths_list': [], 'pix_amps_list': [], 'wvl_widths_list': [], 'detected_wvl_list': [],
    }
    if os.path.isfile(ref_indices_output):
        ref_indices_dict = json_dict_from_file(ref_indices_output)
    else:
        ref_indices_dict = {}
    ref_indices_dict[band] = {}

    def filtered_peaks(peaks_array, sigma=3):
        """
        Filters identiied peaks array based on peak width

        Parameters
        ----------
        peaks_array : np.array
            Generated by find_peaks function
        sigma : int, float
            Distance in sigma allowed from median width
        Returns
        -------

        """
        filtered_peaks_array = peaks_array.copy()
        # try:
        widths = peaks_array[:, 1]
        # except IndexError:
        #     return peaks_array
        width_median = np.median(widths)
        width_dev = np.std(widths)
        width_max = width_median + width_dev * sigma
        width_min = width_median - width_dev * sigma
        sigma_indices = np.asarray(np.nonzero(
            np.logical_and(filtered_peaks_array <= width_max, filtered_peaks_array >= width_min)
        ))[0]  # FIXME: get rid of line duplication
        return filtered_peaks_array[sigma_indices, :]

    for order, spec, wavelengths in zip(orders, specs, map_wavelengths_array):
        try:
            peaks = np.asarray(find_peaks(np.asarray(spec), sigma=peak_finder_sigma))
            peaks = peaks[np.logical_and(peaks[:,0]<domain_ending_pixel, peaks[:,0] > domain_starting_pixel)]
            peaks, ref_indices_groups = curve_fit_peak(
                spec, peaks, wavelengths, line_wavelengths, plt_peak=plt_peak, manual_filter_peak=manual_filter_peak,
                polyfit=polyfit,
                order=order,
                plt_pix=plt_pix, plt_wvl=plt_wvl, line_descriptions=line_descriptions
            )

            if False:  # sigma_filter:  # FIXME: sigma_filter option creates duplicate lines for some reason
                peaks = filtered_peaks(peaks)
            pixpos_array = peaks[:, 0]
            pixwidths_array = peaks[:, 1]
            pixamps_array = peaks[:, 2]
            wvl_array = peaks[:, 4]
            ref_indices_array = peaks[:, 5]
            wvlwidths_array = peaks[:, 6]
            detected_wvl_array = peaks[:, 7]
            mask = np.logical_and(ref_indices_array != 0, ref_indices_array != line_wavelengths.shape[0])
            print(mask)
            json_dict['orders'].append(order)
            json_dict['wvl_list'].append(wvl_array[mask].astype(float).tolist())
            json_dict['ref_indices_list'].append(ref_indices_array[mask].astype(int).tolist())
            json_dict['pixpos_list'].append(pixpos_array[mask].astype(float).tolist())
            json_dict['pix_widths_list'].append(pixwidths_array[mask].astype(float).tolist())
            json_dict['wvl_widths_list'].append(wvlwidths_array[mask].astype(float).tolist())
            json_dict['pix_amps_list'].append(pixamps_array[mask].astype(float).tolist())
            json_dict['detected_wvl_list'].append(detected_wvl_array[mask].astype(float).tolist())
            ref_indices_dict[band][str(order)] = ref_indices_groups

        except IndexError:
            json_dict['orders'].append(order)
            json_dict['wvl_list'].append([])
            json_dict['ref_indices_list'].append([])
            json_dict['pixpos_list'].append([])
            json_dict['pix_widths_list'].append([])
            json_dict['wvl_widths_list'].append([])
            json_dict['pix_amps_list'].append([])
            json_dict['detected_wvl_list'].append([])
            ref_indices_dict[band][str(order)] = []

    save_dict_to_json(json_dict, output_file)
    save_dict_to_json(ref_indices_dict, ref_indices_output)


def repair_identified_lines(identified_lines_file, lines_dat_file):
    line_wavelengths, line_descriptions, line_intensities = load_lines_dat(lines_dat_file)
    # line_wavelengths = lines[:, 0] / 10000
    output_indices = []
    output_wvls = []
    identified_lines_dict = json_dict_from_file(identified_lines_file)
    ref_indices = identified_lines_dict['wvl_widths_list']
    for ref_indices_list in ref_indices:
        ref_indices_array = np.asarray(ref_indices_list).astype(np.int)
        wvls_array = line_wavelengths[ref_indices_array]
        output_indices.append(ref_indices_array.tolist())
        output_wvls.append(wvls_array.tolist())
    identified_lines_dict['wvl_list'] = output_wvls
    identified_lines_dict['ref_indices_list'] = output_indices
    save_dict_to_json(identified_lines_dict, identified_lines_file)


def id_lines_ref_indices_pattern_matching(
        input_id_lines, input_ref_indices, lines_dat_file,
        output_id_lines, output_ref_indices, output_oned_wavemap,
        band, oned_wavemap=None, oned_spec=None, p_pickle=None, centroid_solutions_json_file=None,
        domain_starting_index=4, domain_starting_pixel=0, domain_ending_pixel=None, pixels_in_order=4096,
        gemini_lines_csv=None
):
    if p_pickle is not None:
        polyfit = load_polyfit(p_pickle)
    else:
        polyfit = None
    input_df = load_fitdata_df(
        [input_id_lines], [input_ref_indices], band, centroid_solutions_json_file, domain_starting_index,
        domain_starting_pixel, domain_ending_pixel, pixels_in_order
    )
    orders = input_df.order.unique()
    wavelength_dict = {}
    if oned_wavemap is not None:
        oned_wavemap_json = json_dict_from_file(oned_wavemap)
        for order, spec in zip(oned_wavemap_json['orders'], oned_wavemap_json['specs']):
            wavelength_dict[order] = spec
    else:
        for order in orders:
            wavelength_dict[order] = None
    oned_spec_dict = {}
    if oned_spec is not None:
        oned_spec_json = json_dict_from_file(oned_spec)
        for order, spec in zip(oned_spec_json['orders'], oned_spec_json['specs']):
            oned_spec_dict[order] = spec
    else:
        for order in orders:
            oned_spec_dict[order] = None
    gemini_lines = None
    if gemini_lines_csv is not None:
        gemini_lines = np.loadtxt(gemini_lines_csv, delimiter=',')
        gemini_lines[:, 0] = gemini_lines[:, 0]/10000

    line_wavelengths, line_descriptions, line_intensities = load_lines_dat(lines_dat_file)

    output_oned_wavemap_dict = {'specs': [], 'orders': []}
    for order in orders:
        save = False
        _polyfit = polyfit
        wavelengths = wavelength_dict[order]
        print("order:", order)
        while not save:
            order_df = input_df[input_df.order == order]
            pixels = order_df['pixels'].values
            # standard_devs = np.asarray([0.0001 for i in pixels])
            standard_devs = 0.0001
            all_pixels = np.arange(0, pixels_in_order)
            order_spec = oned_spec_dict[order]
            if wavelengths is None and _polyfit is None:
                reprompt = True
                while reprompt:
                    pixel_domain = input('pixel_domain? pixel1 pixel2 ...')
                    wavelength_range = input('wavelength_range? wavelength1 wavelength2 ...')
                    try:
                        pixel_domain = [float(pixel) for pixel in pixel_domain.split()]
                        wavelength_range = [float(wvl) for wvl in wavelength_range.split()]
                        reprompt = False
                    except (ValueError, TypeError):
                        print("didn't work, please retry")
                A = np.vstack([pixel_domain, np.ones(len(pixel_domain))]).T
                m, y0 = np.linalg.lstsq(A, wavelength_range)[0]
                print(m, y0)
                wavelengths = m * np.arange(0, pixels_in_order) + y0
            calc_wvls, _ = calculate_wvls(pixels, standard_devs, np.asarray(wavelengths), order, polyfit=_polyfit)
            all_wvls, _ = calculate_wvls(all_pixels, standard_devs, np.asarray(wavelengths), order, polyfit=_polyfit)
            min_wvl = all_wvls.min()
            max_wvl = all_wvls.max()
            poly_wvl_to_pix = np.polyfit(all_wvls, all_pixels, 1)
            poly_pix_to_wvl = np.polyfit(all_pixels, all_wvls, 1)

            def forward(wvl):
                return np.polyval(poly_wvl_to_pix, wvl)

            def inverse(pix):
                # return np.interp(pix, all_pixels, all_wvls)
                return np.polyval(poly_pix_to_wvl, pix)

            fig, axs = plt.subplots(1, 1, constrained_layout=True, squeeze=False)
            ax = axs[0, 0]
            ax_twin = ax.twiny()
            # ax_twin.set_xscale('function', functions=(forward, inverse))
            # ax.invert_xaxis()

            cmap = mpl.cm.get_cmap('tab10').colors
            matched_wvls = []
            for i, (row_pixel, row_ref_indices, calc_wvl) in enumerate(zip(order_df['pixels'], order_df['ref_indices'], calc_wvls)):
                print('id_line row pixel:', row_pixel)
                if gemini_lines is None:
                    color = cmap[i%10]
                    ax.plot([row_pixel, row_pixel], [0, 1.1], linestyle='-', color=color)
                    ref_wvl = line_wavelengths[row_ref_indices]
                    matched_wvls.append(ref_wvl)
                    ax_twin.plot([ref_wvl, ref_wvl], [0, 1.0], linestyle='--', color=color)
            if gemini_lines is None:
                plt_wvls = line_wavelengths[np.logical_and(line_wavelengths>min_wvl, line_wavelengths<max_wvl)]
                for wvl in plt_wvls:
                    # if wvl not in matched_wvls:
                    # wvl_pixel = forward(wvl)
                    # ax.plot([wvl_pixel, wvl_pixel], [0, 0.9], linestyle=':', color='black')
                    ax_twin.plot([wvl, wvl], [0, 0.9], linestyle=':', color='black')
            else:
                plt_lines = gemini_lines[np.logical_and(gemini_lines[:,0]>min_wvl, gemini_lines[:,0]<max_wvl)]
                colors = ['red', 'black']
                print('wavelengths:')
                for line in plt_lines:
                    wvl, intensity, identified = line
                    ax_twin.plot([wvl, wvl], [0, intensity], linestyle=':', color=colors[int(identified)])
                    print(wvl, int(identified))
            if order_spec is not None:
                ax2 = ax.twinx()
                # ax2.set_yscale('log')
                ax2.set_xlim(0, pixels_in_order)
                ax2.plot(range(len(order_spec)), order_spec, '.', color='red')
            ax.set_xlim(pixels_in_order, 0)
            print(inverse(pixels_in_order), inverse(0))
            ax_twin.set_xlim(inverse(pixels_in_order), inverse(0))
            plt.show()
            reprompt = True
            while reprompt:
                keep_matches = input('Keep setup (Y/N)?')
                if keep_matches.upper().startswith('Y'):
                    output_oned_wavemap_dict['specs'].append(all_wvls.tolist())
                    output_oned_wavemap_dict['orders'].append(order)
                    save = True
                    reprompt = False
                elif keep_matches.upper().startswith('N'):
                    _polyfit = None
                    wavelengths = None
                    reprompt = False
    save_dict_to_json(output_oned_wavemap_dict, output_oned_wavemap)
    # TODO:
    #  update the 'ref_indices' to be the nearest line
    #  save new output_id_lines
    #  save new output_reference_indices



def gen_echellogram(order_map_file, oned_wavemap_file, output_file, aggregation_axis=0, aggregation=np.nanmean):
    """
    Creates echellogram calibration json file using 1D wavemap file and 2D order_map

    Parameters
    ----------
    order_map_file : str, path
        Path to fits file containing order map
    oned_wavemap_file : str, path
        1D spectrum file containing wavelength corresponding to pixel location, typically generated by gen_oned_spec
        from 2D wavemap file
    output_file : str, path
        Path to output json file
    aggregation_axis : integer, 0 or 1, optional
        numpy array axis to aggregate along
    aggregation : function, optional
        Aggregation function to convert 2D to 1D. np.nanmedian, np.nanmode, np.nanmean, np.nanmax, etc

    Returns
    -------

    """
    order_map_image = ExistingImage(order_map_file).image
    wavelengths, orders = parse_oned_spec(oned_wavemap_file)
    json_dict = {
        'wvl_list': wavelengths, 'x_list': [np.arange(len(wave)).tolist() for wave in wavelengths], 'y_list': [],
        'orders': orders
    }
    if aggregation_axis == 0:
        y_index_image = np.asarray([np.arange(order_map_image.shape[0]) for i in range(order_map_image.shape[1])])
    else:
        y_index_image = np.asarray([np.arange(order_map_image.shape[1]) for i in range(order_map_image.shape[0])])
    if aggregation_axis == 0:
        y_index_image = y_index_image.transpose()
    for order in orders:
        single_order = single_order_image_from_map(y_index_image, order, order_map_image)
        oned_spec = aggregation(single_order, axis=aggregation_axis)
        oned_spec_no_nan = fill_in_nan(oned_spec)
        json_dict['y_list'].append(oned_spec_no_nan.astype(float).tolist())

    save_dict_to_json(json_dict, output_file)


def gen_error_dict(fit_polynomial, fitdata_df, pickle_output_file):
    fit_wvl = fit_polynomial(fitdata_df['pixels'], fitdata_df['order'])
    fit_wvl = fit_wvl / fitdata_df['order']
    error = fitdata_df['wavelength'] - fit_wvl
    standard_error = np.sqrt(np.sum(error ** 2) / (error.shape[0] - 1))
    print("{} error: {}".format(pickle_output_file, standard_error))
    fit_dict = {
        'pixpos': fitdata_df['pixels'].tolist(),
        'order': fitdata_df['order'].tolist(),
        'fit_wvl': fit_wvl.tolist(),
        'identified_lines_wvl': fitdata_df['wavelength'].tolist(),
        'error': error.tolist(),
        'standard_error': standard_error,
        'pickle_filename': pickle_output_file,
    }
    return fit_dict


def fitdata_df_to_identified_lines(fitdata_df, identified_lines_fname_prefix):
    ref_names_unique = fitdata_df['ref_name'].unique()
    save_names = []
    for ref_name in ref_names_unique:
        ref_name_df = fitdata_df[fitdata_df['ref_name'] == ref_name]
        orders_unique = ref_name_df['order'].unique()
        save_dict = {'orders': [], 'pixpos_list': [], 'wvl_list': [], 'ref_indices_list': [], 'ref_name': ref_name}
        for order in orders_unique:
            save_dict['orders'].append(int(order))
            order_df = ref_name_df[ref_name_df['order'] == order]
            save_dict['pixpos_list'].append(order_df['pixels'].tolist())
            save_dict['wvl_list'].append(order_df['wavelength'].tolist())
            save_dict['ref_indices_list'].append(order_df['ref_indices'].tolist())
        save_name = identified_lines_fname_prefix + ref_name + '.json'
        save_names.append(save_name)
        save_dict_to_json(save_dict, save_name)
    return save_names


def reconcile_id_lines_and_ref_indices(identified_lines_json_files, ref_indices_json_files, band):
    for ident_lines, ref_indices in zip(identified_lines_json_files, ref_indices_json_files):
        id_lines_dict = json_dict_from_file(ident_lines)
        ref_indices_dict = json_dict_from_file(ref_indices)
        for order, lines in zip(id_lines_dict['orders'], id_lines_dict['ref_indices_list']):
            ref_indices_new = []
            ref_indices_old = ref_indices_dict[band][str(order)]
            for ref_ind_list in ref_indices_old:
                ref_ind_list_new = []
                for ref_ind in ref_ind_list:
                    if ref_ind in lines:
                        ref_ind_list_new.append(ref_ind)
                if ref_ind_list_new:
                    ref_indices_new.append(ref_ind_list_new)
            ref_indices_dict[band][str(order)] = ref_indices_new
        save_dict_to_json(ref_indices_dict, os.path.join(os.path.dirname(ident_lines), os.path.basename(ref_indices)))


def remove_high_error_lines(
        fit_pickle, fit_output_file, fit_error_dict, error_sigma, identified_lines_json_files, ref_indices_json_files,
        band,
        centroid_solutions_json_file=None, domain_starting_index=4, domain_starting_pixel=0, domain_ending_pixel=None,
        pixels_in_order=4096
):
    with open(fit_pickle, 'rb') as f:
        p = pickle.load(f)
    fitdata_df = load_fitdata_df(
        identified_lines_json_files, ref_indices_json_files, band, centroid_solutions_json_file,
        domain_starting_index, domain_starting_pixel, domain_ending_pixel, pixels_in_order
    )
    print('load dataframe, shape', fitdata_df.shape)
    err_dict = gen_error_dict(p, fitdata_df, fit_pickle)
    print('gen err_dict')
    save_dict_to_json(err_dict, fit_output_file)
    print('save err_dict')
    print('about to load', fit_error_dict)
    fit_dict = json_dict_from_file(fit_error_dict)
    print('load fit_dict')
    error_array = np.asarray(err_dict['error'])
    print('get error_array', error_array)
    std_error = fit_dict['standard_error']
    print('std_error', std_error)
    large_error = np.where(np.abs(error_array) > error_sigma * std_error)
    print('bool large_error')
    fitdata_df = fitdata_df.drop(fitdata_df.index[large_error])
    print('drop large_error, dataframe shape', fitdata_df.shape)
    low_error_identified_lines_prefix = os.path.join(
        os.path.dirname(fit_pickle), 'low_error', 'identified_lines.{}.low_error.'.format(band)
    )
    print('prefix', low_error_identified_lines_prefix)
    low_error_identified_lines = fitdata_df_to_identified_lines(fitdata_df, low_error_identified_lines_prefix)
    print(low_error_identified_lines)
    reconcile_id_lines_and_ref_indices(low_error_identified_lines, ref_indices_json_files, band)
    print('reconciled ref_indices')


def gen_echellogram_fit_wvlsol(
    echellogram_json_file, identified_lines_json_files, ref_indices_json_files, output_file, pixels_in_order,
    centroid_solutions_json_file=None, domain_starting_index=0, fit_output_file='fit.json',
    pixel_degree=4, order_degree=3, p_init_pickle=None, pickle_output_file=None, band='YJ', sigma=2,
    domain_starting_pixel=0, domain_ending_pixel=None
):
    """
    Creates echellogram calibration json file using wavelength solution

    Parameters
    ----------
    echellogram_json_file : str, path
        Echellogram file from gen_echellogram, used for 'y_list' values only
    identified_lines_json_files : list of str, path
        identified lines calibration file from gen_identified_lines
    ref_indices_json_files : list of str, path
        reference indices calibration file from gen_reference_indices
    output_file : str, path
        Path to output json file
    pixels_in_order : int
        pixels along order, 2048 for H2RG and 4096 for H4RG
    centroid_solutions_json_file : str, path, optional
        *.centroid_solutions.json file created from pipeline flat step
    domain_starting_index : int, optional
        starting domain value from centroid solutions file, 3 to skip the pipeline generated extra
    fit_output_file : str, path, optional
        json file detailing the fit parameters used for the echellogram
    pixel_degree : int, optional
        fitting degrees for pixels
    order_degree : int, optional
        fitting degrees for orders
    p_init_pickle : str, path, optional
        optional fitting pickle file to use as initial guess for echellogram fitting, must be
        of type astropy.modeling.polynomial.PolynomialBase
    pickle_output_file : str, path, optional
        output file for wavelength solution polynomial
    band : str, optional
        band from ref_indices json file
    sigma : int, float, optional
        error cutoff before refitting
    domain_starting_pixel: int
    domain_ending_pixel: int, None

    Returns
    -------

    """
    # TODO: eliminate echellogram json file, and use centroid solutions values for y_list instead
    fitdata_df = load_fitdata_df(
        identified_lines_json_files, ref_indices_json_files, band, centroid_solutions_json_file, domain_starting_index,
        domain_starting_pixel, domain_ending_pixel, pixels_in_order
    )
    if p_init_pickle is not None:
        with open(p_init_pickle, 'rb') as f:
            p_init = pickle.load(f)
    else:
        p_init = None
    p, fit_results = fit_wvlsol(fitdata_df, pixels_in_order, pixel_degree, order_degree, p_init=p_init)

    # save_dict_to_json({'p':p, 'fit_results': fit_results}, fit_output_file)
    assert isinstance(p, PolynomialBase)
    if pickle_output_file is None:
        pickle_output_file = output_file.replace('.json', '.p')

    fit_dict = gen_error_dict(p, fitdata_df, pickle_output_file)
    error_array = np.asarray(fit_dict['error'])
    std_error = fit_dict['standard_error']
    large_error = np.where(np.abs(error_array)>sigma*std_error)
    print(fitdata_df.loc[large_error])
    fitdata_df = fitdata_df.drop(fitdata_df.index[large_error])
    p, fit_results = fit_wvlsol(fitdata_df, pixels_in_order, pixel_degree, order_degree, p_init=p_init)

    fit_dict = gen_error_dict(p, fitdata_df, pickle_output_file)
    save_dict_to_json(fit_dict, fit_output_file)
    fitdata_df_to_identified_lines(
        fitdata_df,
        os.path.join(os.path.dirname(output_file), 'low_error', 'identified_lines.{}.low_error.'.format(band))
    )

    with open(pickle_output_file, 'wb') as f:
        pickle.dump(p, f)
    json_dict = {
        'wvl_list': [], 'x_list': [], 'y_list': [], 'orders': [],
    }
    pixels = np.arange(0, pixels_in_order)
    for order in fitdata_df['order'].unique():
        order_int = int(order)
        p_out = p(pixels, np.asarray([order_int for i in range(pixels_in_order)]))
        wvl = p_out / order_int
        json_dict['orders'].append(order_int)
        json_dict['x_list'].append(pixels.tolist())
        json_dict['wvl_list'].append(wvl.tolist())
    json_dict['y_list'] = json_dict_from_file(echellogram_json_file)['y_list']
    save_dict_to_json(json_dict, output_file)


def gen_ref_indices(
        identified_lines_json_file, lines_dat_file, band_name, updated_identified_lines_output, ref_indices_output
):
    """
    Generates the reference indices json calibration file

    Parameters
    ----------
    identified_lines_json_file : str, path
        identified lines json file generated by gen_identified_lines
    lines_dat_file : str, path
        lines dat file used to create identified_lines_json_file
    band_name : str
        band calibtration is being generated for
    updated_identified_lines_output : str, path
        output identified lines json file name with line degeneracies fixed
    ref_indices_output : str, path
        output reference indices json file name
    Returns
    -------

    """
    if os.path.isfile(ref_indices_output):
        ref_dict = json_dict_from_file(ref_indices_output)
    else:
        ref_dict = {}
    id_lines = json_dict_from_file(identified_lines_json_file)
    lines = np.loadtxt(lines_dat_file)
    intensities = lines[:, 1]
    wavelengths = lines[:, 0]
    indices = np.arange(*intensities.shape)
    dtype = [('index', np.int), ('wavelength', np.float64), ('intensity', np.float64)]
    lines = np.asarray(list(zip(indices, wavelengths, intensities)), dtype=dtype)
    new_id_lines_dict = {
        'orders': [], 'pixpos_list': [], 'ref_indices_list': [], 'ref_name': os.path.basename(lines_dat_file),
        'wvl_list': id_lines['wvl_list']
    }
    band_dict = {}
    for order, pix_pos, ref_index in zip(id_lines['orders'], id_lines['pixpos_list'], id_lines['ref_indices_list']):
        ref_index_array = np.asarray(ref_index)
        ref_index_array_new = ref_index_array.copy()
        try:
            unq, count = np.unique(ref_index_array, axis=0, return_counts=True)
            # repeat_index = unq[count>1]
            # repeat_index_count = count[count>1]
        except ValueError:
            unq = np.asarray([])
            count = np.asarray([])
        ref_indices = []
        for index, index_count in zip(unq, count):
            if index_count == 1:
                ref_indices.append([int(index)])
            else:
                # for index, index_count in zip(repeat_index, repeat_index_count):
                wavelength = wavelengths[index]
                max_index = index + index_count
                min_index = index - index_count
                initial_matches = lines[min_index:max_index].copy()
                initial_matches['wavelength'] = abs(initial_matches['wavelength'] - wavelength)
                sorted_matches = np.sort(initial_matches, order='wavelength')
                sorted_matches_cutoff = sorted_matches[0:index_count]
                resorted_matches = np.sort(sorted_matches_cutoff, order='index')
                new_indices = resorted_matches['index']
                ref_index_array_new[ref_index_array==index] = new_indices
                ref_indices.append(new_indices.tolist())

        band_dict[str(order)] = ref_indices
        new_id_lines_dict['orders'].append(order)
        new_id_lines_dict['pixpos_list'].append(pix_pos)
        new_id_lines_dict['ref_indices_list'].append(ref_index_array_new.tolist())

    ref_dict[band_name] = band_dict
    save_dict_to_json(ref_dict, ref_indices_output)
    save_dict_to_json(new_id_lines_dict, updated_identified_lines_output)


def index_matching(list1, list2):
    """
    Generates indices necessary to match 2 lists based on values

    Parameters
    ----------
    list1 : list
    list2 : list

    Returns
    -------

    """
    if len(list1) != len(list2):
        raise ValueError('lists must be of the same length')
    index_list = []
    for v1 in list1:
        for i2, v2 in enumerate(list2):
            if v1 == v2:
                index_list.append(i2)
                continue
    return np.asarray(index_list)


def get_orders_from_map(order_map_image):
    """
    Get unique values from order map

    Parameters
    ----------
    order_map_image : np.array

    Returns
    -------

    """
    orders = np.unique(order_map_image)
    orders = orders[np.logical_and(orders.astype(int) == orders, orders > 0)]
    return orders


def parse_oned_spec(oned_spec_json_file):
    oned_spec = json_dict_from_file(oned_spec_json_file)
    return oned_spec['specs'], oned_spec['orders']


def gen_oned_wavemap(order_map_file, wavemap_file, output_file, aggregation_axis=0):
    gen_oned_spec(order_map_file, wavemap_file, output_file, aggregation_axis=aggregation_axis)


def json_dict_from_file(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def save_dict_to_json(dictionary, filename):
    json_string = json.dumps(dictionary)
    try:
        with open(filename, 'w') as f:
            f.write(json_string)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as f:
            f.write(json_string)


def fill_in_nan(oned_array, maskzeros=True, return_linear_fit_array=False):
    if maskzeros:
        nanmask = np.logical_or(np.isnan(oned_array), oned_array == 0)
    else:
        nanmask = np.isnan(oned_array)
    x = np.arange(*oned_array.shape)
    y = oned_array.copy()
    slope, intercept, r_value, p_value, std_err = linregress(x[~nanmask], y[~nanmask])
    if return_linear_fit_array:
        y = slope * x + intercept
    else:
        y[nanmask] = slope * x[nanmask] + intercept
    return y


def single_order_image_from_map(image, order, order_map_image):
    single_order = np.empty(order_map_image.shape)
    single_order[:] = np.nan
    single_order[order_map_image == order] = image[order_map_image == order]
    return single_order


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def plot_echellogram_error(fit_json_file):
    plot_with_order_legend(fit_json_file, 'identified_lines_wvl', x_axis_label='Library wavelength', x_axis_units='microns')
    plot_with_order_legend(fit_json_file, 'order')
    plot_with_order_legend(fit_json_file, 'pixpos', x_axis_label='column')
    # plot_by_order(fit_json_file, 'pixpos', x_axis_label='column')


def plot_with_order_legend(fit_json_file, x_axis_key, x_axis_label=None, x_axis_units=None):
    plt.rcParams.update({'font.size': 18})
    _f = plt.figure()
    _f.set_figwidth(15)
    _f.set_figheight(10)
    fit_dict = json_dict_from_file(fit_json_file)
    df = pd.DataFrame(fit_dict)
    orders = df.order.unique()
    order_df = OrderedDict()

    resolution_element = fit_dict['standard_error'] * 3000

    if x_axis_label is None:
        x_axis_label = x_axis_key
        # x_axis_label = 'Library wavelength'

    if x_axis_units is not None:
        x_axis_label += ' ({})'.format(x_axis_units)

    for order in orders:
        order_df[order] = df.loc[df.order == order]
        plt.scatter(order_df[order][x_axis_key], order_df[order]['error'], label=order)

    # plt.title(fit_json_file.replace('.json', '') + ' error_res_elem_frac={:.3f}'.format(resolution_element))
    plt.title('{} Residuals Between Library and Lamp Line Wavelengths for Kr,Ar,Xe,Hg'.format(spectral_band))
    plt.xlabel(x_axis_label)
    plt.ylabel('error (microns)')
    error_max = df['error'].max()
    error_min = df['error'].min()

    if error_max > 0:
        error_max = 1.1 * error_max
    else:
        error_max = 0.9 * error_max
    if error_min < 0:
        error_min = 1.1 * error_min
    else:
        error_min = 0.9 * error_min
    plt.legend(loc='lower right')
    plt.ylim(bottom=error_min, top=error_max)
    plt.xlim(left=df[x_axis_key].min()-0.1, right=df[x_axis_key].max()+0.2)
    plt.savefig(fit_json_file.replace('json', '{}.png'.format(x_axis_key)))
    plt.show()


def plot_by_order(fit_json_file, x_axis_key, x_axis_label=None, x_axis_units=None):
    fit_dict = json_dict_from_file(fit_json_file)
    df = pd.DataFrame(fit_dict)
    orders = df.order.unique()
    order_df = OrderedDict()

    resolution_element = fit_dict['standard_error'] * 3000

    if x_axis_label is None:
        x_axis_label = x_axis_key

    if x_axis_units is not None:
        x_axis_label += ' ({})'.format(x_axis_units)

    for order in orders:
        plt.rcParams.update({'font.size': 16})
        _f = plt.figure()
        _f.set_figwidth(15)
        _f.set_figheight(10)
        order_df[order] = df.loc[df.order == order]
        plt.scatter(order_df[order][x_axis_key], order_df[order]['error'], label=order)

        plt.title(fit_json_file.replace('.json', '') + ' error_res_elem_frac={:.3f}'.format(resolution_element))
        plt.xlabel(x_axis_label)
        plt.ylabel('error (microns)')
        error_max = df['error'].max()
        error_min = df['error'].min()

        if error_max > 0:
            error_max = 1.1 * error_max
        else:
            error_max = 0.9 * error_max
        if error_min < 0:
            error_min = 1.1 * error_min
        else:
            error_min = 0.9 * error_min
        plt.legend()
        plt.ylim(bottom=error_min, top=error_max)
        plt.savefig(fit_json_file.replace('json', '{}.{}.png'.format(order, x_axis_key)))
        plt.show()


def gen_even_spaced_lines_dat_file(output_filename, spacing=50, beginning_wvl_A=6000, end_wvl_A=40000, intensity=100):
    wvls = np.arange(beginning_wvl_A, end_wvl_A, spacing)
    intensities = intensity * np.ones(wvls.shape)
    output_data = np.stack((wvls, intensities), axis=1)
    fmt = ('%1.3f', '%1.3e')
    np.savetxt(output_filename, output_data, delimiter=' ', fmt=fmt)


def gen_even_spaced_lines_csv_file(
        output_filename, spacing=0.0050, beginning_wvl_u=0.6000, end_wvl_u=4.0000, intensity=100
):
    wvls = np.arange(beginning_wvl_u, end_wvl_u, spacing)
    intensities = intensity * np.ones(wvls.shape)
    output_data = np.stack((wvls, intensities), axis=1)
    fmt = '%1.18e'
    np.savetxt(output_filename, output_data, delimiter=';', fmt=fmt)


def plot_oned_spec(
        oned_spec_file, identified_lines_file, fit_pickle, title='', cutoff=0, pixel_domain_start=0,
        pixel_domain_end=None
):
    oned_spec = json_dict_from_file(oned_spec_file)
    identified_lines = json_dict_from_file(identified_lines_file)
    # wavelengths_all = []
    # intensities_all = []
    with open(fit_pickle, 'rb') as f:
        fit = pickle.load(f)
    for order, intensities in zip(oned_spec['orders'], oned_spec['specs']):
        intensities_arr = np.asarray(intensities)
        intensity_cutoff = np.where(intensities_arr > cutoff)
        intensities_arr = intensities_arr[intensity_cutoff]
        pixels = np.arange(len(intensities))[intensity_cutoff]
        pixels_cutoff = np.logical_and(pixels>pixel_domain_start, pixels<pixel_domain_end)
        pixels = pixels[pixels_cutoff]
        intensities_arr = intensities_arr[pixels_cutoff]
        order_array = np.ones(pixels.shape) * order
        wavelengths = (fit(pixels, order_array) / order)
        # wavelengths_all += wavelengths
        # intensities_all += intensities
        plt.plot(wavelengths, intensities_arr, label=str(order))
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Intensity (ADU)')
        plt.title(title)
        plt.legend()
    plt.show()


def plt_resolving_power(identified_lines_json):
    plt.rcParams.update({'font.size': 30})
    _f = plt.figure()
    _f.set_figwidth(15)
    _f.set_figheight(10)
    _dict = json_dict_from_file(identified_lines_json)
    fwhm_conversion_factor = 4 * np.sqrt(np.log(2))  # TODO fix width in gen_identified_lines and remove this factor
    wvls = _dict['wvl_list']
    wvl_widths = _dict['wvl_widths_list']
    wvls_flat = np.asarray([item for sublist in wvls for item in sublist])
    wvl_widths_flat = np.asarray([item for sublist in wvl_widths for item in sublist]) * fwhm_conversion_factor
    resolving_power = wvls_flat/wvl_widths_flat
    plt.scatter(wvls_flat, resolving_power, label='each detected line')

    # median_wvls = np.asarray([np.median(sublist) for sublist in wvls])
    # median_widths = np.asarray([np.median(sublist) for sublist in wvl_widths]) * fwhm_conversion_factor
    # resolving_power = median_wvls/median_widths
    # plt.scatter(median_wvls, resolving_power, label='median by spectral order')
    # plt.legend()
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Resolving Power')
    plt.show()
    print('resolving power', np.median(resolving_power))
    print('resolving power dev', np.std(resolving_power))


def recipe_parser(recipe_file):
    _obsdate = recipe_file.split(os.path.sep)[-1]
    _obsdate = _obsdate.split('.')[-2]
    _bands = ('YJ', 'HK')
    _file_format = _obsdate + '.rimas.{:04d}.{}.fits'
    with open(recipe_file) as f:
        _lines = f.readlines()
    _dicts = []
    _lines = [_line for _line in _lines if not _line.startswith('#')]
    _keys = [_s.strip() for _s in _lines[0].split(',')]
    for _line in _lines[1:]:
        _values = [_s.strip() for _s in _line.split(',')]
        _dict = {}
        for _k, _v in zip(_keys, _values):
            _dict[_k.upper()] = _v
        frametypes = _dict['FRAMETYPES'].split()
        frametypes = [_t.upper()=='ON' or _t.upper()=='A' for _t in frametypes]
        _file_nums = _dict['OBSIDS'].split()
        _file_nums = [int(_s) for _s in _file_nums]
        _yj_files = [_file_format.format(_file_num, 'YJ') for _file_num in _file_nums]
        _hk_files = [_file_format.format(_file_num, 'HK') for _file_num in _file_nums]
        _name = _dict['OBJNAME']
        _save_name_yj = _name + '.YJ.fits'
        _save_name_hk = _name + '.HK.fits'
        _dicts.append(
            {
                'name': _name,
                'save_name_yj': _save_name_yj,
                'save_name_hk': _save_name_hk,
                'pos_neg': np.asarray(frametypes),
                'files_yj': np.asarray(_yj_files),
                'files_hk': np.asarray(_hk_files)
            }
        )
    return _dicts


def recipe_reduce(recipe_file, data_dir, _reduce_dir=None):
    if _reduce_dir is None:
        _reduce_dir = os.path.join(data_dir, 'reduced')
    _recipe_dicts = recipe_parser(recipe_file)
    for _recipe in _recipe_dicts:
        for _band in ('yj', 'hk'):
            _save_name = os.path.join(_reduce_dir, _recipe['save_name_{}'.format(_band)])
            _pos_files = [os.path.join(data_dir, _f) for _f in _recipe['files_{}'.format(_band)][_recipe['pos_neg']]]
            _neg_files = [os.path.join(data_dir, _f) for _f in _recipe['files_{}'.format(_band)][np.logical_not(_recipe['pos_neg'])]]
            _pos_images = [ExistingImage(_f).image for _f in _pos_files]
            _neg_images = [ExistingImage(_f).image for _f in _neg_files]
            _res_image = np.median(np.asarray(_pos_images), axis=0) - np.median(np.asarray(_neg_images), axis=0)
            ArrayImage(_res_image).save(_save_name)


def update_ref_indices(old_dat, new_dat, ref_indices, identified_lines):
    old_dat_wvl = load_lines_dat(old_dat)[0]
    new_dat_wvl = load_lines_dat(new_dat)
    translation_dict = {}
    for i, wvl in enumerate(old_dat_wvl):
        translation_dict[i] = find_nearest(new_dat_wvl, wvl)[1]
    ref_indices_dict = json_dict_from_file(ref_indices)
    identified_lines_dict = json_dict_from_file(identified_lines)
    new_ref_indices = {}
    for band, band_dict in ref_indices_dict.items():
        new_ref_indices[band] = {}
        for order, index in band_dict.items():
            new_ref_indices[band][order] = [[translation_dict[i] for i in lst] for lst in index]
    identified_lines_ref_indices = identified_lines_dict['ref_indices_list']
    new_identified_lines_ref_indices = [[translation_dict[i] for i in lst] for lst in identified_lines_ref_indices]
    identified_lines_dict['ref_indices_list'] = new_identified_lines_ref_indices
    new_dat_basename = os.path.basename(new_dat)
    identified_lines_dict["ref_name"] = new_dat_basename
    save_dict_to_json(identified_lines_ref_indices, identified_lines.replace('.json', new_dat_basename+'.json'))
    save_dict_to_json(new_ref_indices, ref_indices.replace('.json', new_dat_basename+'.json'))


if __name__ == '__main__':
    run_recipe_reduce = False
    run_gen_oned_spec = False
    run_gen_oned_maps = False
    run_gen_identified_lines = False
    run_gen_echellogram = True
    run_gen_echellogram_fit_wvlsol = True
    run_gen_ref_indices = False
    run_remove_high_error_lines = False
    run_plot_error = False
    run_plot_oned_spec = False
    run_plot_residuals = False
    run_id_lines_ref_indices_pattern_matching = False
    # observation_date = 20230104
    observation_date = 20230726
    slit_width = 80
    pix_deg = 3
    order_deg = 3
    # RIMAS files
    spectral_band = 'HK'
    band_domain = {
        'YJ': (1240, 2200),
        'HK': (1500, 3330)
    }
    pixel_start, pixel_end = band_domain[spectral_band]
    order_map = r'..\indata\{}\{}-micron-grism-maps\{}_order_map_extended.fits'.format(
        observation_date, slit_width, spectral_band)
    wavemap = r'..\indata\{}\{}-micron-grism-maps\{}_wavmap_extended.fits'.format(
        observation_date, slit_width, spectral_band)
    # xenon - mercury - argon
    # spectrum = r''
    flat_spectrum = r'..\indata\{}\{}-micron-grism-reduced\cals-{}.{}.fits'.format(
        observation_date, slit_width, spectral_band.lower(), spectral_band)
    # order_map = r'G:\My Drive\RIMAS\RIMAS spectra\modeled_spectra\rimas_h4rg\rollover-removed\HK_order_map_extended.fits'
    # wavemap = r'G:\My Drive\RIMAS\RIMAS spectra\modeled_spectra\rimas_h4rg\rollover-removed\HK_wavmap_extended.fits'
    # # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20210304\ohlines\ohlines.fits'
    # # spectrum = r'C:\Users\durba\Documents\echelle\simulations\20210316\even_spaced_25-stuermer-1000s.fits'
    # # spectrum = r'C:\Users\durba\Documents\echelle\simulations\20210316\even_spaced_10.fits'
    # # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20201008\ohlines\ohlines.fits'
    # # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20201008\rimas.0026.YJ.C0.fits'
    # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20210513\rimas.0026.YJ.C0.fits'
    # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\20220622\on-off-subtracted\xenon-mercury-argon.HK.fits'
    # ohline_dat = r'C:\Users\durba\PycharmProjects\plp\master_calib\igrins\ohlines.dat'
    elements = [
        # 'Xe',
        # 'Hg',
        # 'Ar',
        # 'Kr',
        # 'Ne',
        # 'HgAr',
        # 'XeHgAr',
        # 'XeHgArKr',
        'arc'
    ]
    elements_dict = {
        'Xe': 'xenon',
        'Hg': 'mercury',
        'Ar': 'argon',
        'Kr': 'krypton',
        'Ne': 'mercury',
        'XeHgAr': 'xenon-mercury-argon',
        # 'XeHgArKr': 'xenon-argon-krypton-mercury',
        'XeHgArKr': 'xenon-mercury-argon-krypton',
        'arc': 'xenon-mercury-argon-krypton'
    }
    elements_str = ''.join(elements)
    # elements_str = 'Kr'
    try:
        element = elements_dict[elements_str]
    except KeyError:
        element = ''
    # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\{}\reduced\{}.{}.fits'.format(
    #     observation_date, element, spectral_band)
    spectrum = r'..\indata\{}\{}-micron-grism-reduced\{}.{}.fits'.format(
        observation_date, slit_width, element, spectral_band)
    spectrum_type = 'arc' + '.' + elements_str
    ohline_dat = r'..\master_calib\rimas\{}_lines.dat'.format(elements_str)
    element_dats = [r'..\master_calib\rimas\{}_lines.dat'.format(e) for e in elements]
    combine_dats = [r'..\master_calib\rimas\{}_lines.dat'.format(e) for e in ['Xe', 'Ar']]
    # combine_lines_dat(combine_dats, ohline_dat)
    # # ohline_dat = 'even_spaced_25.dat'
    # # ohline_dat = 'even_spaced_10.dat'
    # centroid_solutions_file = r'..\calib\primary\20201008\FLAT_rimas.0000.YJ.C0.centroid_solutions.json'
    #
    # # output_dir = 'pickle_fit_test_med_oh'
    # # output_dir = 'even_spaced_25-stuermer'
    # # output_dir = 'even_spaced_10-stuermer'
    # output_dir = 'rimas_h4rg_arc_comb_low_error'
    output_dir = 'rimas_h4rg_{}_micron'.format(slit_width)

    # DeVeny files

    # output_dir = 'deveny'
    # order_map = 'deveny_order_map.fits'  # TODO
    # wavemap = 'deveny_wavemap.fits'  # TODO
    # spectrum = '20210506.0014.fits'
    # ohline_dat = 'CdArNeHg_lines.dat'  # TODO


    # output_dir = os.path.join(output_dir, 'pickle_fit__no_repeats')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # skyline_output_filename = os.path.join(output_dir, 'YJ_oned.json')
    skyline_output_format = os.path.join(output_dir, '{}.{}_oned.json')
    skyline_output_filename = skyline_output_format.format(spectrum_type, spectral_band)
    flat_output_filename = os.path.join(output_dir, '{}.{}_oned.json'.format('flat', spectral_band))
    # wavemap_output_filename = 'YJ_oned_wavemap.json'
    # wavemap_output_filename = os.path.join(output_dir, 'YJ_oned_wavemap_linear_fit.json')
    # identified_lines_output_filename = os.path.join(output_dir, 'YJ_identified_lines.json')
    # echellogram_output_file = os.path.join(output_dir, 'YJ_echellogram.json')
    wavemap_output_filename = os.path.join(output_dir, '{}_oned_wavemap_linear_fit.json'.format(spectral_band))
    identified_lines_output_format = os.path.join(output_dir, '{}.{}_identified_lines.curvefit_peaks.json')
    # for element in elements:
    #     repair_identified_lines(
    #         identified_lines_output_format.format('arc.{}'.format(element), spectral_band),
    #         r'C:\Users\durba\PycharmProjects\plp\master_calib\rimas\{}_lines.dat'.format(element)
    #     )
    identified_lines_output_filename = identified_lines_output_format.format(spectrum_type, spectral_band)
    echellogram_output_file = os.path.join(output_dir, '{}.{}_echellogram.json'.format(spectral_band, elements_str))
    ref_indices_output_format = os.path.join(output_dir, 'ref_{}lines_indices.json')
    ref_indices_output_file = ref_indices_output_format.format(spectrum_type)
    updated_identified_lines_output_filename = identified_lines_output_filename.replace('.json', 'update.json')
    curve_fit_echellogram_output_filename = echellogram_output_file.replace('.json', '_curvefit.json')
    fit_wvlsol_echellogram_output_filename = echellogram_output_file.replace(
        '.json', '_multiple_id_lines_curvefit_peaks_fit_wvlsol__p{}_o{}.json'
    )
    # p_init_pickle_filename = echellogram_output_file.replace('.json', 'fit_wvlsol__p{}_o{}.json')
    # p_init_pickle_filename = r'C:\Users\durba\PycharmProjects\plp\master_calib_creation\rimas_h4rg_arc_comb\HK.XeHgArKr_echellogram_fit_wvlsol__p4_o3.p'
    # even_spaced_dat = 'even_spaced_10.dat'
    # even_spaced_csv = even_spaced_dat.replace('dat', 'csv')
    fit_wvlsol_echellogram_output_filename = fit_wvlsol_echellogram_output_filename.format(pix_deg, order_deg)
    fit_wvlsol_pickle_output_filename = fit_wvlsol_echellogram_output_filename.replace('.json', '.p')
    fit_output_filename = '.{}.fit_p{}m{}-domain.json'
    fit_output_filename = fit_output_filename.format(spectral_band, pix_deg, order_deg)
    fit_output_filename = fit_wvlsol_echellogram_output_filename.replace('.json', fit_output_filename)
    fit_wvlsol_pickle_init_dict = {
        # 'HK': None,
        # 'YJ': None,
        'HK': r'rimas_h4rg_80_micron\HK.XeHgArKr_echellogram_multiple_id_lines_curvefit_peaks_fit_wvlsol__p3_o3.p',
        'YJ': r'rimas_h4rg_80_micron\YJ.Ar_echellogram_multiple_id_lines_curvefit_peaks_fit_wvlsol__p3_o3.p'
    }
    fit_wvlsol_pickle_init_filename = fit_wvlsol_pickle_init_dict[spectral_band]
    # p_init_pickle_filename = p_init_pickle_filename.format(pix_deg, order_deg)
    # p_init_pickle_filename = p_init_pickle_filename.replace('.XeHgArKr', '')
    # file_overlay(order_map, spectrum)
    # file_overlay(wavemap, spectrum)
    # file_overlay(order_map, wavemap)

    # gen_even_spaced_lines_dat_file(even_spaced_dat, spacing=10)
    # gen_even_spaced_lines_csv_file(even_spaced_csv, spacing=0.0010)
    if run_recipe_reduce:
        data_date = str(observation_date)
        reduce_dir = r'..\indata\{}'.format(data_date)
        recipe_dir = r'..\recipe_logs'
        recipe_reduce(os.path.join(recipe_dir, data_date+'.recipes'), reduce_dir)
    if run_gen_oned_spec:
        gen_oned_spec(order_map, spectrum, skyline_output_filename, 0)
        # gen_oned_spec(order_map, flat_spectrum, flat_output_filename, 0)
    if run_gen_oned_maps:
        gen_oned_spec(order_map, wavemap, wavemap_output_filename, 0, np.nanmax)
    if run_gen_identified_lines:
        print(identified_lines_output_filename)
        gen_identified_lines(
            skyline_output_filename, wavemap_output_filename, ohline_dat, identified_lines_output_filename,
            ref_indices_output_file, spectral_band,
            p_init_pickle=fit_wvlsol_pickle_init_filename,
            # p_init_pickle=os.path.join(output_dir, 'argon_pattern_wvl_sol.p'),
            plt_peak=True,
            manual_filter_peak=True,
            domain_starting_pixel=pixel_start, domain_ending_pixel=pixel_end,
            sigma_filter=False,
            # plt_wvl=True,
            plt_pix=True
        )

    if run_id_lines_ref_indices_pattern_matching:
        id_lines_ref_indices_pattern_matching(
            identified_lines_output_filename, ref_indices_output_file, ohline_dat,
            identified_lines_output_filename.replace('.json', '.pattern.json'),
            ref_indices_output_file.replace('.json', '.pattern.json'),
            wavemap_output_filename.replace('.json', '.pattern.json'),
            spectral_band, oned_wavemap=wavemap_output_filename, oned_spec=skyline_output_filename,
            p_pickle=os.path.join(output_dir, 'argon_pattern_wvl_sol.p'),
            centroid_solutions_json_file=None,
            domain_starting_index=4, domain_starting_pixel=0, domain_ending_pixel=None, pixels_in_order=4096,
            # gemini_lines_csv=r'C:\Users\durba\Downloads\Ar_combined.csv'
        )

    if run_gen_echellogram:
        gen_echellogram(order_map, wavemap_output_filename, echellogram_output_file, 0, np.nanmean)
    if run_gen_ref_indices:
        gen_ref_indices(
            identified_lines_output_filename, ohline_dat, spectral_band,
            updated_identified_lines_output_filename, ref_indices_output_file
        )
    if run_gen_echellogram_fit_wvlsol:
        gen_echellogram_fit_wvlsol(
            echellogram_output_file,
            [identified_lines_output_format.format('arc.{}'.format(element), spectral_band) for element in elements],
            # updated_identified_lines_output_filename,
            [ref_indices_output_format.format('arc.{}'.format(element)) for element in elements],
            fit_wvlsol_echellogram_output_filename, 4096,
            # centroid_solutions_file, 3,
            fit_output_file=fit_output_filename,
            pixel_degree=pix_deg, order_degree=order_deg, pickle_output_file=fit_wvlsol_pickle_output_filename,
            band=spectral_band, sigma=3, domain_starting_pixel=pixel_start, domain_ending_pixel=pixel_end
            # p_init_pickle=fit_wvlsol_pickle_init_filename
        )
    if run_remove_high_error_lines:
        # fit_pickle, fit_output_file, fit_error_dict, error_sigma, identified_lines_json_files, ref_indices_json_files,
        # band,
        # centroid_solutions_json_file = None, domain_starting_index = 4, domain_starting_pixel = 0, domain_ending_pixel = None,
        # pixels_in_order = 4096
        remove_high_error_lines(
            fit_wvlsol_pickle_init_filename,
            identified_lines_output_format.format('fit.{}'.format(element), spectral_band),
            r'rimas_h4rg_arc_comb_low_error\fit_p3m3-domain.YJ.json', 3,
            [identified_lines_output_format.format('arc.{}'.format(element), spectral_band) for element in elements],
            [ref_indices_output_format.format('arc.{}'.format(element)) for element in elements], spectral_band, None,
            3, pixel_start, pixel_end, 4096
        )
    if run_plot_error:
        print(fit_output_filename)
        plot_echellogram_error(fit_output_filename)
    if run_plot_oned_spec:
        for element in elements:
            print(element)
            oned = skyline_output_format.format('arc.{}'.format(element), spectral_band)
            print(oned)
            id_lines = identified_lines_output_format.format('arc.{}'.format(element), spectral_band)
            plot_oned_spec(
                oned, id_lines, fit_wvlsol_pickle_output_filename,
                '{} gas lamps'.format(spectral_band), pixel_domain_start=pixel_start, pixel_domain_end=pixel_end
            )
        # plot_oned_spec(
        #     flat_output_filename, identified_lines_output_filename, fit_wvlsol_pickle_output_filename,
        #     '{} flat'.format(spectral_band), pixel_domain_start=pixel_start, pixel_domain_end=pixel_end
        # )
    # plt_resolving_power(identified_lines_output_filename)
