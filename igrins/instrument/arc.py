import os

import astropy.units as u
from astropy.table import vstack
from astroquery.nist import Nist
from astroquery.atomic import AtomicLineList
import numpy as np

config = {
    'elements': [
        'Xe',
        'Hg',
        'Ar',
        'Kr',
        # 'Ne'
    ],
    'wavelength_range': (0.800, 1.00),
    'wavelength_error': 0.0008,
    'units': u.micron,
    'bands': ['HK', 'YJ'],  # ['ArI'],
    'wavelength_steps': 0.1
}


def load_lines_dat(dat_filename):
    try:
        lines = np.loadtxt(dat_filename, dtype='float,float,int,U12', usecols=(0, 1, 2, 3))
        line_wavelengths = [line[0] / 10000 for line in lines]
        line_intensities = [line[1] for line in lines]
        line_descriptions = ['{}: {}-{}'.format(i, line[3][0:2], line[2]) for i, line in enumerate(lines)]
    except (IndexError, ValueError):
        lines = np.loadtxt(dat_filename).transpose()
        line_wavelengths = lines[0]/10000
        line_intensities = lines[1]
        line_descriptions = [str(i) for i in range(line_wavelengths.shape[0])]
    return np.asarray(line_wavelengths), np.asarray(line_descriptions), line_intensities


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
        lines = q['Observed']
        _mask = lines.mask
        lines[_mask] = q['Ritz'][_mask]
        print(lines.shape)
        # lines = lines[~lines.mask]
        lines_dict[element] = lines.data
    return lines_dict


def get_lines_atomic(config_dict=config, wavelength_accuracy=1/1000):
    lines_dict = {}

    wavelength_ends = np.arange(
        config_dict['wavelength_range'][0], config_dict['wavelength_range'][1], config_dict['wavelength_steps']
    )
    wavelength_ends = [wvl * config_dict['units'] for wvl in wavelength_ends]

    for element in config_dict['elements']:
        query_dict = dict(wavelength_type='Vacuum', wavelength_accuracy=wavelength_accuracy, element_spectrum=element)
        start_wavelength_range = (wavelength_ends[0], wavelength_ends[1])
        q = AtomicLineList.query_object(wavelength_range=start_wavelength_range, **query_dict)
        for i, _wvl in enumerate(wavelength_ends[1:-1]):
            wavelength_range = (_wvl, wavelength_ends[1:][i+1])
            q = vstack([q, AtomicLineList.query_object(wavelength_range=wavelength_range, **query_dict)])
        lines = q['LAMBDA VAC ANG']
        # _mask = lines.mask
        # lines[_mask] = q['Ritz'][_mask]
        print(lines.shape)
        # lines = lines[~lines.mask]
        lines_dict[element] = lines.data
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
    dat_array = []
    wav_array = []
    name_array = []
    intensity_array = []
    index_array = []
    for dat in dat_iterable:
        print(dat)
        line_wavelengths, line_descriptions, line_intensities = load_lines_dat(dat)
        line_wavelengths = line_wavelengths * 10000
        # _data = np.loadtxt(dat)
        _indices = np.arange(line_wavelengths.shape[0])
        _name = os.path.basename(dat)
        _names = np.asarray([_name for i in _indices])
        wav_array.append(line_wavelengths)
        _data = np.rec.fromarrays((line_wavelengths, line_intensities, _indices, _names))
        dat_array.append(_data)
        name_array.append(_names)
        intensity_array.append(line_intensities)
        index_array.append(_indices)

    data = np.concatenate(dat_array, axis=0)
    waves = np.concatenate(wav_array, axis=0)
    _indices = np.concatenate(index_array, axis=0)
    _names = np.concatenate(name_array, axis=0)
    _intensities = np.concatenate(intensity_array, axis=0)
    waves_sort = waves.argsort()

    data = data[waves_sort]
    _indices = _indices[waves_sort]
    _names = _names[waves_sort]
    _intensities = _intensities[waves_sort]
    waves = waves[waves_sort]

    unique_waves = np.unique(waves)

    combined_line_data = []
    _combined_intensities = []
    _combined_names = []
    _combined_index = []

    for wave in unique_waves:
        _slice = waves == wave
        _intensity = np.sum(_intensities[_slice])
        _max_int = np.max(_intensities[_slice])
        _max_int_index = np.where(_intensities[_slice] == _max_int)[0][0]
        _name = _names[_slice][_max_int_index]
        _index = _indices[_slice][_max_int_index]
        combined_line_data.append((wave, _intensity, _index, _name))
        _combined_intensities.append(_intensity)
        _combined_names.append(_name)
        _combined_index.append(_index)

    # print(data)
    # data_df = pd.DataFrame({'wavelength': data[0], 'intensity': data[1]})
    # data_df = data_df.groupby('wavelength').sum()
    # print(data_df)
    # dat_array = data_df.to_numpy().transpose()
    # print(dat_array)
    # error = config['wavelength_error']
    # wavelength_unique = np.unique(data[0])
    # intensities = np.asarray([np.sum(data[1][np.logical_and(data[0]<(wav+error), data[0]>(wav-error))]) for wav in wavelength_unique])
    # np.savetxt(output_dat_filename, data, fmt=('%.3f', '%.3e','%d', '%s'))
    save_array = np.rec.fromarrays((unique_waves, _combined_intensities, _combined_index, _combined_names))
    np.savetxt(output_dat_filename, save_array, fmt=('%.3f', '%.3e', '%d', '%s'))


if __name__ == '__main__':
    bands = ['HK', 'YJ']
    csv_dir = r'C:\Users\durba\Downloads'
    dat_dir = "..\\..\\master_calib\\rimas"
    # for _element, wavelengths in get_lines_atomic().items():
    #     element_str = _element.split()[0]
    #     for band in bands:
    #         csv_file = '{band}_{element}_spectrum.csv'.format(band=band, element=element_str)
    #         csv_file = os.path.join(csv_dir, csv_file)
    #         intensities = get_intensities(csv_file, wavelengths)
    #         dat_file = "{band}_{element}_lines.dat".format(band=band, element=element_str)
    #         dat_file = os.path.join(dat_dir, dat_file)
    #         save_lines(dat_file, wavelengths, intensities)
    input_dat_template = os.path.join(dat_dir, "{band}_{element}_lines.dat")
    element_dat_template = os.path.join(dat_dir, "{element}_lines.dat")
    final_dat = element_dat_template.format(element='arc')
    # for element in config['elements']:
    #     element_str = element.split()[0]
    #     combine_lines_dat(
    #         [input_dat_template.format(band=band, element=element_str) for band in config['bands']],
    #         element_dat_template.format(element=element_str)
    #     )
    element_str = ''.join([e.split()[0] for e in config['elements']])
    combine_lines_dat(
        [element_dat_template.format(element=element) for element in config['elements']],
        final_dat
    )
