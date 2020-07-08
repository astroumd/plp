import os

import astropy.units as u
from astroquery.nist import Nist
import numpy as np
import pandas as pd

config = {
    'elements': ['Xe', 'Hg', 'Ar', 'Kr'],
    'wavelength_range': (0.800, 2.500),
    'wavelength_error': 0.0008,
    'units': u.micron,
    'bands': ['HK', 'YJ']
}


def get_lines(config_dict=config):
    lines_dict = {}
    for element in config_dict['elements']:
        q = Nist.query(
            config_dict['wavelength_range'][0]*config_dict['units'],
            config_dict['wavelength_range'][1]*config_dict['units'],
            element,
            output_order='wavelength',
            wavelength_type='vacuum',
        )
        lines = q['Observed'].data
        print(lines.shape)
        # lines = lines[~lines.mask]
        lines_dict[element] = lines
    return lines_dict


def get_intensities(spectrum_csv, lines, config_dict=config):
    spectrum = np.loadtxt(spectrum_csv, delimiter=',')
    spectrum = spectrum.transpose()
    print(spectrum)
    error = config_dict['wavelength_error']
    intensities = []
    lines = lines/10000
    print(lines)
    for line in lines:
        in_range = np.logical_and(spectrum[0]<(line+error), spectrum[0]>(line-error))
        intensities.append(np.sum(spectrum[1][in_range]))
    return np.asarray(intensities)


def save_lines(dat_filename, wavelength_array, intensity_array):
    save_array = np.asarray([wavelength_array, intensity_array]).transpose()
    np.savetxt(dat_filename, save_array, fmt=('%.3f', '%.3e'))


def combine_lines_dat(dat_iterable, output_dat_filename):
    dat_array = [np.loadtxt(dat) for dat in dat_iterable]
    data = np.concatenate(dat_array, axis=0).transpose()
    # print(data)
    # data_df = pd.DataFrame({'wavelength': data[0], 'intensity': data[1]})
    # data_df = data_df.groupby('wavelength').sum()
    # print(data_df)
    # dat_array = data_df.to_numpy().transpose()
    # print(dat_array)
    error = config['wavelength_error']
    wavelength_unique = np.unique(data[0])
    intensities = np.asarray([np.sum(data[1][np.logical_and(data[0]<(wav+error), data[0]>(wav-error))]) for wav in wavelength_unique])
    save_lines(output_dat_filename, wavelength_unique, intensities)


if __name__ == '__main__':
    # bands = ['HK', 'YJ']
    # csv_dir = r'C:\Users\durba\Downloads'
    dat_dir = "..\\..\\master_calib\\rimas"
    # for element, wavelengths in get_lines().items():
    #     for band in bands:
    #         csv_file = '{band}_{element}_spectrum.csv'.format(band=band, element=element)
    #         csv_file = os.path.join(csv_dir, csv_file)
    #         intensities = get_intensities(csv_file, wavelengths)
    #         dat_file = "{band}_{element}_lines.dat".format(band=band, element=element)
    #         dat_file = os.path.join(dat_dir, dat_file)
    #         save_lines(dat_file, wavelengths, intensities)
    input_dat_template = os.path.join(dat_dir, "{band}_{element}_lines.dat")
    element_dat_template = os.path.join(dat_dir, "{element}_lines.dat")
    final_dat = element_dat_template.format(element='arc')
    for element in config['elements']:
        combine_lines_dat(
            [input_dat_template.format(band=band, element=element) for band in config['bands']],
            element_dat_template.format(element=element)
        )
    combine_lines_dat(
        [element_dat_template.format(element=element) for element in config['elements']],
        final_dat
    )
