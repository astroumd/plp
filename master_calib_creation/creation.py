import json
import os

import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

from master_calib_creation.image import ExistingImage, file_overlay

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


def gen_echellogram_curve_fit(
    echellogram_json_file, identified_lines_json_file, output_file, pixels_in_order,
    centroid_solutions_json_file=None, domain_starting_index=0, fit_output_file='fit.json'
    # , pixel_degree, order_degree
):
    identified_lines = json_dict_from_file(identified_lines_json_file)
    num_orders = len(identified_lines['orders'])
    indices = range(num_orders)
    if centroid_solutions_json_file is not None:
        domains = json_dict_from_file(centroid_solutions_json_file)['domain']
        domains = domains[domain_starting_index:domain_starting_index+num_orders]
    else:
        domains = [list(range(pixels_in_order)) for j in indices]

    fitdata = [[],[],[]]

    for j in indices:
        domain = domains[j]
        wvls = identified_lines['wvl_list'][j]
        pixpos = identified_lines['pixpos_list'][j]
        wvls_array = np.asarray(wvls)
        pixpos_array = np.asarray(pixpos)
        domain_indices = np.logical_and(pixpos_array>domain[0], pixpos_array<domain[1])
        pixpos = pixpos_array[domain_indices].tolist()
        wvls = wvls_array[domain_indices].tolist()
        order = identified_lines['orders'][j]
        order_list = [order for i in range(len(wvls))]
        fitdata = [fitdata[0]+wvls, fitdata[1]+pixpos, fitdata[2]+order_list]
    fitdata_array = np.asarray(fitdata)

    def wavelength(c_array, pix_array, m):
        (
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            # c20, c21, c22, c23,
            # c30, c31, c32, c33
        ) = c_array
        p = pix_array
        # return c00 + c01 * m + c02 * m ** 2 + \
        #        c10 * p + c11 * p * m + c12 * p * m ** 2 + \
        #        c20 * p ** 2 + c21 * p ** 2 * m + c22 * p ** 2 * m ** 2  # + \
        #        c30 * p ** 3 + c31 * p ** 3 * m + c32 * p ** 3 * m ** 2
        return c00 + c01*m + c02*m**2 + c03*m**3 +\
               c10*p + c11*p*m + c12*p*m**2 + c13*p*m**3  # + \
        #        c20*p**2 + c21*p**2*m + c22*p**2*m**2 + c23*p**2*m**3 +\
               # c30*p**3 + c31*p**3*m + c32*p**3*m**2 + c33*p**3*m**3

    def func(data,
             c00, c01, c02, c03,
             c10, c11, c12, c13,
             # c20, c21, c22, c23,
             # c30, c31, c32, c33
             ):
        constants = (
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            # c20, c21, c22, c23,
            # c30, c31, c32, c33
        )
        return wavelength(constants, data[1], data[2])

    # guess = (
    #     1,1,1,1,
    #     1,1,1,1,
    #     1,1,1,1,
    #     1,1,1,1
    # )
    params, pcov = curve_fit(func, fitdata_array, fitdata_array[0])  # , guess)
    pixels = np.arange(0, pixels_in_order)
    json_dict = {
        'wvl_list': [], 'x_list': [], 'y_list': [], 'orders': [],
    }
    for order in identified_lines['orders']:
        json_dict['orders'].append(order)
        json_dict['x_list'].append(pixels.tolist())
        json_dict['wvl_list'].append(wavelength(params, pixels, order).tolist())
    json_dict['y_list'] = json_dict_from_file(echellogram_json_file)['y_list']
    save_dict_to_json(json_dict, output_file)

    fit_wvl = wavelength(params, fitdata_array[1], fitdata_array[2])
    error = fitdata_array[0] - fit_wvl
    standard_error = np.sqrt(np.sum(error**2) / (error.shape[0]-1))
    fit_dict = {
        'param_names': [
            'c00', 'c01', 'c02', 'c03',
            'c10', 'c11', 'c12', 'c13',
            # 'c20', 'c21', 'c22', 'c23',
            # 'c30', 'c31', 'c32', 'c33'
        ],
        'params': params.tolist(),
        'pixpos': fitdata_array[1].tolist(),
        'order': fitdata_array[2].tolist(),
        'fit_wvl': fit_wvl.tolist(),
        'identified_lines_wvl': fitdata_array[0].tolist(),
        'error': error.tolist(),
        'standard_error': standard_error
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


def gen_ref_indices_alt1(identified_lines_json_file, band_name, ref_indices_output):
    if os.path.isfile(ref_indices_output):
        ref_dict = json_dict_from_file(ref_indices_output)
    else:
        ref_dict = {}
    id_lines = json_dict_from_file(identified_lines_json_file)
    band_dict = {}
    for order, indices in zip(id_lines['orders'], id_lines['ref_indices_list']):
        band_dict[str(order)] = [indices]
    ref_dict[band_name] = band_dict
    save_dict_to_json(ref_dict, ref_indices_output)


def gen_ref_indices_alt2(identified_lines_json_file, band_name, ref_indices_output):
    if os.path.isfile(ref_indices_output):
        ref_dict = json_dict_from_file(ref_indices_output)
    else:
        ref_dict = {}
    id_lines = json_dict_from_file(identified_lines_json_file)
    band_dict = {}
    for order, indices in zip(id_lines['orders'], id_lines['ref_indices_list']):
        band_dict[str(order)] = [[_index] for _index in indices]
    ref_dict[band_name] = band_dict
    save_dict_to_json(ref_dict, ref_indices_output)


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
    from matplotlib import pyplot as plt

    fit_dict = json_dict_from_file(fit_json_file)
    plt.plot(fit_dict['order'], fit_dict['error'], 'bo')
    plt.title(fit_json_file.replace('.json', ''))
    plt.xlabel('order')
    plt.ylabel('error')
    plt.show()


if __name__ == '__main__':
    order_map = r'G:\My Drive\RIMAS\RIMAS spectra\modeled_spectra\echelle\YJ_order_map_extended.fits'
    wavemap   = r'G:\My Drive\RIMAS\RIMAS spectra\modeled_spectra\echelle\YJ_wavmap_extended.fits'
    spectrum = r'G:\My Drive\RIMAS\RIMAS spectra\echelle simulator\simulations\20201008\rimas.0026.YJ.C0.fits'
    ohline_dat = r'C:\Users\durba\PycharmProjects\plp\master_calib\igrins\ohlines.dat'
    skyline_output_filename = 'YJ_oned.json'
    # wavemap_output_filename = 'YJ_oned_wavemap.json'
    wavemap_output_filename = 'YJ_oned_wavemap_linear_fit.json'
    identified_lines_output_filename = 'YJ_identified_lines.json'
    echellogram_output_file = 'YJ_echellogram.json'
    ref_indices_output_file = 'ref_ohlines_indices.json'
    fit_output_filename = 'fit_p2m2.json'
    centroid_solutions_file = r'..\calib\primary\20201008\FLAT_rimas.0000.YJ.C0.centroid_solutions.json'
    updated_identified_lines_output_filename = identified_lines_output_filename.replace('.json', 'update.json')
    curve_fit_echellogram_outpout_filename = echellogram_output_file.replace('.json', '_curvefit.json')
    # file_overlay(order_map, spectrum)
    # file_overlay(wavemap, spectrum)
    # file_overlay(order_map, wavemap)
    # gen_oned_spec(order_map, spectrum, skyline_output_filename, 1)
    # gen_oned_spec(order_map, wavemap, wavemap_output_filename, 1, np.nanmax)
    # gen_identified_lines(skyline_output_filename, wavemap_output_filename, ohline_dat, identified_lines_output_filename)
    # gen_echellogram(order_map, wavemap_output_filename, echellogram_output_file, 1, np.nanmean)
    # gen_ref_indices(
    #     identified_lines_output_filename, ohline_dat, 'YJ',
    #     updated_identified_lines_output_filename, ref_indices_output_file
    # )

    # gen_ref_indices_alt1(updated_identified_lines_output_filename, 'YJ', 'single_list'+ref_indices_output_file)
    # gen_ref_indices_alt2(updated_identified_lines_output_filename, 'YJ', 'individual_lists'+ref_indices_output_file)
    # gen_echellogram_curve_fit(
    #     echellogram_output_file, updated_identified_lines_output_filename, curve_fit_echellogram_outpout_filename, 2048,
    #     centroid_solutions_file, 3, fit_output_filename
    # )
    plot_echellogram_error(fit_output_filename)
