import json
import os
from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd
from astropy.modeling.polynomial import Chebyshev2D
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

from master_calib_creation.image import ExistingImage, file_overlay

from igrins.procedures.find_peak import find_peaks
from igrins.procedures.process_derive_wvlsol import fit_wvlsol


def gen_oned_spec(order_map_file, twod_spectrum_file, output_file, aggregation_axis=0, aggregation=np.nanmedian):

    order_map_image = ExistingImage(order_map_file, fits_image_hdu=0).image
    spectrum_image = ExistingImage(twod_spectrum_file, fits_image_hdu=0).image
    gen_oned_spec_image(
        order_map_image, spectrum_image, output_file, aggregation_axis=aggregation_axis, aggregation=aggregation
    )


def gen_oned_spec_image(order_map_image, spectrum_image, output_file, aggregation_axis=0, aggregation=np.nanmedian):
    orders = get_orders_from_map(order_map_image)
    json_dict = {'specs': [], 'orders': []}
    for order in orders:
        single_order = single_order_image_from_map(spectrum_image, order, order_map_image)
        # single_order = np.empty(order_map_image.shape)
        # single_order[:] = np.nan
        # single_order[order_map_image == order] = spectrum_image[order_map_image == order]
        oned_spec = aggregation(single_order, axis=aggregation_axis)
        oned_spec_no_nan = fill_in_nan(oned_spec)
        # oned_spec_no_nan = np.nan_to_num(oned_spec.copy())
        json_dict['specs'].append(oned_spec_no_nan.astype(float).tolist())
        json_dict['orders'].append(int(order))
    save_dict_to_json(json_dict, output_file)


def gen_identified_lines(oned_spec_json_file, oned_wavemap_file, lines_dat_file, output_file):
    specs, orders = parse_oned_spec(oned_spec_json_file)
    map_wavelengths, wavemap_orders = parse_oned_spec(oned_wavemap_file)
    order_indices = index_matching(orders, wavemap_orders)
    map_wavelengths_array = np.asarray(map_wavelengths)[order_indices]
    lines = np.loadtxt(lines_dat_file)
    line_wavelengths = lines[:, 0] / 10000
    json_dict = {
        'wvl_list': [], 'ref_name': os.path.basename(lines_dat_file), 'ref_indices_list': [], 'pixpos_list': [],
        'orders': []
    }

    def filtered_peaks(peaks_array, sigma=3):
        filtered_peaks_array = peaks_array.copy()
        widths = peaks_array[:, 1]
        width_median = np.median(widths)
        width_dev = np.std(widths)
        width_max = width_median + width_dev * sigma
        width_min = width_median - width_dev * sigma
        sigma_indices = np.asarray(np.nonzero(
            np.logical_and(filtered_peaks_array <= width_max, filtered_peaks_array >= width_min)
        ))[0]
        return filtered_peaks_array[sigma_indices, :]

    def line_lookup(detected_peaks):
        line_waves = []
        line_index = []
        for peak in detected_peaks:
            value, index = find_nearest(line_wavelengths, peak)
            line_waves.append(value)
            line_index.append(index)
        return np.asarray(line_waves), np.asarray(line_index)

    for order, spec, wavelengths in zip(orders, specs, map_wavelengths_array):
        # wavelengths_nonzero = wavelengths[np.nonzero(wavelengths)]
        # ref_indices_array = np.asarray(np.nonzero(
        #     np.logical_and(line_wavelengths <= wavelengths_nonzero.max(), line_wavelengths >= wavelengths_nonzero.min())
        # ))[0]
        # wvl_array = line_wavelengths[ref_indices_array]
        # pixpos_array = np.interp(wvl_array, np.arange(wavelengths.shape[0]), wavelengths)
        peaks = filtered_peaks(np.asarray(find_peaks(np.asarray(spec))))
        # peaks = filtered_peaks(peaks)
        pixpos_array = peaks[:, 0]
        detected_wvl_array = np.interp(pixpos_array, np.arange(wavelengths.shape[0]), wavelengths)
        wvl_array, ref_indices_array = line_lookup(detected_wvl_array)
        # pixpos_array = np.interp(wvl_array, wavelengths, np.arange(wavelengths.shape[0]))
        json_dict['orders'].append(order)
        json_dict['wvl_list'].append(wvl_array.astype(float).tolist())
        json_dict['ref_indices_list'].append(ref_indices_array.astype(int).tolist())
        json_dict['pixpos_list'].append(pixpos_array.astype(float).tolist())

    save_dict_to_json(json_dict, output_file)


def gen_echellogram(order_map_file, oned_wavemap_file, output_file, aggregation_axis=0, aggregation=np.nanmean):
    order_map_image = ExistingImage(order_map_file).image
    wavelengths, orders = parse_oned_spec(oned_wavemap_file)
    json_dict = {
        'wvl_list': wavelengths, 'x_list': [np.arange(len(wave)).tolist() for wave in wavelengths], 'y_list': [],
        'orders': orders
    }
    y_index_image = np.asarray([np.arange(2048) for i in range(2048)])
    if aggregation_axis == 0:
        y_index_image = y_index_image.transpose()
    for order in orders:
        single_order = single_order_image_from_map(y_index_image, order, order_map_image)
        oned_spec = aggregation(single_order, axis=aggregation_axis)
        oned_spec_no_nan = fill_in_nan(oned_spec)
        json_dict['y_list'].append(oned_spec_no_nan.astype(float).tolist())

    save_dict_to_json(json_dict, output_file)


def gen_echellogram_fit_wvlsol(
    echellogram_json_file, identified_lines_json_file, ref_indices_json_file, output_file, pixels_in_order,
    centroid_solutions_json_file=None, domain_starting_index=0, fit_output_file='fit.json',
    pixel_degree=4, order_degree=3, p_init_pickle=None, pickle_output_file=None, band='YJ'
):
    identified_lines = json_dict_from_file(identified_lines_json_file)
    ref_indices_dict = json_dict_from_file(ref_indices_json_file)['YJ']
    num_orders = len(identified_lines['orders'])
    indices = range(num_orders)
    if centroid_solutions_json_file is not None:
        domains = json_dict_from_file(centroid_solutions_json_file)['domain']
        domains = domains[domain_starting_index:domain_starting_index+num_orders]
    else:
        domains = [(0, pixels_in_order) for j in indices]

    fitdata = {'pixels':[], 'order':[], 'wavelength':[]}

    for j in indices:
        order = identified_lines['orders'][j]
        domain = domains[j]
        wvls = identified_lines['wvl_list'][j]
        pixpos = identified_lines['pixpos_list'][j]
        ref_indices = identified_lines['ref_indices_list'][j]
        repeat_ref_indices = ref_indices_dict[str(order)]
        repeat_ref_indices = [item for sublist in repeat_ref_indices for item in sublist]  # flattening lists
        wvls_array = np.asarray(wvls)
        pixpos_array = np.asarray(pixpos)
        ref_indices_array = np.asarray(ref_indices)
        if len(repeat_ref_indices) == 0:
            no_repeats_array = np.ones(ref_indices_array.shape, dtype=np.bool)
        else:
            repeat_list = np.asarray([ref_indices_array != i for i in repeat_ref_indices])
            no_repeats_array = np.all(repeat_list, axis=0)
        domain_indices = np.logical_and(pixpos_array>domain[0], pixpos_array<domain[1], no_repeats_array)
        pixpos = pixpos_array[domain_indices].tolist()
        wvls = wvls_array[domain_indices].tolist()
        order_list = [order for i in range(len(wvls))]
        fitdata['wavelength'] = fitdata['wavelength'] + wvls
        fitdata['pixels'] = fitdata['pixels'] + pixpos
        fitdata['order'] = fitdata['order'] + order_list

    fitdata_df = pd.DataFrame(fitdata)
    if p_init_pickle is not None:
        with open(p_init_pickle, 'rb') as f:
            p_init = pickle.load(f)
    else:
        p_init = None
    p, fit_results = fit_wvlsol(fitdata_df, pixel_degree, order_degree, p_init=p_init)

    # save_dict_to_json({'p':p, 'fit_results': fit_results}, fit_output_file)
    assert isinstance(p, Chebyshev2D)
    if pickle_output_file is None:
        pickle_output_file = output_file.replace('.json', '.p')
    with open(pickle_output_file, 'wb') as f:
        pickle.dump(p, f)

    pixels = np.arange(0, pixels_in_order)

    json_dict = {
        'wvl_list': [], 'x_list': [], 'y_list': [], 'orders': [],
    }
    for order in identified_lines['orders']:
        p_out = p(pixels, np.asarray([order for i in range(pixels_in_order)]))
        wvl = p_out / order
        json_dict['orders'].append(order)
        json_dict['x_list'].append(pixels.tolist())
        json_dict['wvl_list'].append(wvl.tolist())
    json_dict['y_list'] = json_dict_from_file(echellogram_json_file)['y_list']
    save_dict_to_json(json_dict, output_file)

    fit_wvl = p(fitdata_df['pixels'], fitdata_df['order'])
    fit_wvl = fit_wvl / fitdata_df['order']
    error = fitdata_df['wavelength'] - fit_wvl
    standard_error = np.sqrt(np.sum(error**2) / (error.shape[0]-1))
    fit_dict = {
        'pixpos': fitdata_df['pixels'].tolist(),
        'order': fitdata_df['order'].tolist(),
        'fit_wvl': fit_wvl.tolist(),
        'identified_lines_wvl': fitdata_df['wavelength'].tolist(),
        'error': error.tolist(),
        'standard_error': standard_error,
        'pickle_filename': pickle_output_file,
    }
    save_dict_to_json(fit_dict, fit_output_file)


def gen_ref_indices(
        identified_lines_json_file, lines_dat_file, band_name, updated_identified_lines_output, ref_indices_output
):
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
        unq, count = np.unique(ref_index_array, axis=0, return_counts=True)
        repeat_index = unq[count>1]
        repeat_index_count = count[count>1]
        ref_indices = []
        for index, index_count in zip(repeat_index, repeat_index_count):
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
    if len(list1) != len(list2):
        raise ValueError('lists must be of the same length')
    index_list = []
    for v1 in list1:
        for v2, i2 in zip(list2, range(len(list2))):
            if v1 == v2:
                index_list.append(i2)
                continue
    return np.asarray(index_list)


def get_orders_from_map(order_map_image):
    orders = np.unique(order_map_image)
    orders = orders[orders > orders.max() / 2]
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
    plot_with_order_legend(fit_json_file, 'identified_lines_wvl', x_axis_label=None, x_axis_units='microns')
    plot_with_order_legend(fit_json_file, 'order')
    plot_with_order_legend(fit_json_file, 'pixpos', x_axis_label='column')
    plot_by_order(fit_json_file, 'pixpos', x_axis_label='column')


def plot_with_order_legend(fit_json_file, x_axis_key, x_axis_label=None, x_axis_units=None):
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


if __name__ == '__main__':
    order_map = r'G:\My Drive\RIMAS\RIMAS spectra\modeled_spectra\echelle\YJ_order_map_extended.fits'
    wavemap   = r'G:\My Drive\RIMAS\RIMAS spectra\modeled_spectra\echelle\YJ_wavmap_extended.fits'
    spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20210304\ohlines\ohlines.fits'
    # spectrum = r'C:\Users\durba\Documents\echelle\simulations\20210316\even_spaced_25-stuermer-1000s.fits'
    # spectrum = r'C:\Users\durba\Documents\echelle\simulations\20210316\even_spaced_10.fits'
    # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20201008\ohlines\ohlines.fits'
    # spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20201008\rimas.0026.YJ.C0.fits'
    ohline_dat = r'C:\Users\durba\PycharmProjects\plp\master_calib\igrins\ohlines.dat'
    # ohline_dat = 'even_spaced_25.dat'
    # ohline_dat = 'even_spaced_10.dat'
    centroid_solutions_file = r'..\calib\primary\20201008\FLAT_rimas.0000.YJ.C0.centroid_solutions.json'

    # output_dir = 'pickle_fit_test_med_oh'
    output_dir = 'even_spaced_25-stuermer'
    # output_dir = 'even_spaced_10-stuermer'
    output_dir = os.path.join(output_dir, 'pickle_fit__no_repeats')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    skyline_output_filename = os.path.join(output_dir, 'YJ_oned.json')
    # wavemap_output_filename = 'YJ_oned_wavemap.json'
    wavemap_output_filename = os.path.join(output_dir, 'YJ_oned_wavemap_linear_fit.json')
    identified_lines_output_filename = os.path.join(output_dir, 'YJ_identified_lines.json')
    echellogram_output_file = os.path.join(output_dir, 'YJ_echellogram.json')
    ref_indices_output_file = os.path.join(output_dir, 'ref_ohlines_indices.json')
    fit_output_filename = os.path.join(output_dir, 'fit_p{}m{}-domain.json')
    updated_identified_lines_output_filename = identified_lines_output_filename.replace('.json', 'update.json')
    curve_fit_echellogram_output_filename = echellogram_output_file.replace('.json', '_curvefit.json')
    fit_wvlsol_echellogram_output_filename = echellogram_output_file.replace('.json', '_fit_wvlsol__p{}_o{}.json')
    # p_init_pickle_filename = 'even_spaced_25-stuermer\\YJ_echellogram.json.p'
    even_spaced_dat = 'even_spaced_10.dat'
    even_spaced_csv = even_spaced_dat.replace('dat', 'csv')
    pix_deg = 3
    order_deg = 5
    fit_output_filename = fit_output_filename.format(pix_deg, order_deg)
    fit_wvlsol_echellogram_output_filename = fit_wvlsol_echellogram_output_filename.format(pix_deg, order_deg)
    fit_wvlsol_pickle_output_filename = fit_wvlsol_echellogram_output_filename.replace('.json', '.p')
    p_init_pickle_filename = 'even_spaced_25-stuermer\\YJ_echellogram_fit_wvlsol__p{}_o{}.p'.format(pix_deg, order_deg)
    # file_overlay(order_map, spectrum)
    # file_overlay(wavemap, spectrum)
    # file_overlay(order_map, wavemap)

    # gen_even_spaced_lines_dat_file(even_spaced_dat, spacing=10)
    # gen_even_spaced_lines_csv_file(even_spaced_csv, spacing=0.0010)
    gen_oned_spec(order_map, spectrum, skyline_output_filename, 1)
    gen_oned_spec(order_map, wavemap, wavemap_output_filename, 1, np.nanmax)
    gen_identified_lines(skyline_output_filename, wavemap_output_filename, ohline_dat, identified_lines_output_filename)
    gen_echellogram(order_map, wavemap_output_filename, echellogram_output_file, 1, np.nanmean)
    gen_ref_indices(
        identified_lines_output_filename, ohline_dat, 'YJ',
        updated_identified_lines_output_filename, ref_indices_output_file
    )
    # gen_echellogram_curve_fit(
    #     echellogram_output_file, updated_identified_lines_output_filename, curve_fit_echellogram_output_filename, 2048,
    #     centroid_solutions_file, 3,
    #     fit_output_file=fit_output_filename
    # )
    #
    gen_echellogram_fit_wvlsol(
        echellogram_output_file, updated_identified_lines_output_filename, ref_indices_output_file,
        fit_wvlsol_echellogram_output_filename, 2048,
        centroid_solutions_file, 3,
        fit_output_file=fit_output_filename,
        pixel_degree=pix_deg, order_degree=order_deg, pickle_output_file=fit_wvlsol_pickle_output_filename,
        p_init_pickle=p_init_pickle_filename
    )

    plot_echellogram_error(fit_output_filename)
