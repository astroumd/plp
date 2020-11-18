import json
import os

import numpy as np
from scipy.stats import linregress

from .image import ExistingImage, file_overlay

from igrins.procedures.find_peak import find_peaks

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
    for order, spec, wavelengths in zip(orders, specs, map_wavelengths_array):
        wavelengths_nonzero = wavelengths[np.nonzero(wavelengths)]
        ref_indices_array = np.asarray(np.nonzero(
            np.logical_and(line_wavelengths <= wavelengths_nonzero.max(), line_wavelengths >= wavelengths_nonzero.min())
        ))[0]
        wvl_array = line_wavelengths[ref_indices_array]
        # pixpos_array = np.interp(wvl_array, np.arange(wavelengths.shape[0]), wavelengths)
        pixpos_array = np.interp(wvl_array, wavelengths, np.arange(wavelengths.shape[0]))
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


if __name__ == '__main__':
    order_map = r'G:\My Drive\RIMAS spectra\modeled_spectra\echelle\YJ_order_map_extended.fits'
    wavemap   = r'G:\My Drive\RIMAS spectra\modeled_spectra\echelle\YJ_wavmap_extended.fits'
    spectrum = r'G:\My Drive\RIMAS spectra\echelle simulator\simulations\20201008\rimas.0026.YJ.C0.fits'
    ohline_dat = r'C:\Users\durba\PycharmProjects\plp\master_calib\igrins\ohlines.dat'
    skyline_output_filename = 'YJ_oned.json'
    # wavemap_output_filename = 'YJ_oned_wavemap.json'
    wavemap_output_filename = 'YJ_oned_wavemap_linear_fit.json'
    identified_lines_output_filename = 'YJ_identified_lines.json'
    echellogram_output_file = 'YJ_echellogram.json'
    # file_overlay(order_map, spectrum)
    # file_overlay(order_map, wavemap)
    # gen_oned_spec(order_map, spectrum, skyline_output_filename, 1)
    # gen_oned_spec(order_map, wavemap, wavemap_output_filename, 1, np.nanmax)
    # gen_identified_lines(skyline_output_filename, wavemap_output_filename, ohline_dat, identified_lines_output_filename)
    gen_echellogram(order_map, wavemap_output_filename, echellogram_output_file, 1, np.nanmean)
