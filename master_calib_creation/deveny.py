from astroquery.nist import Nist
from astropy import units as u
import numpy as np

from master_calib_creation.image import ExistingImage, ArrayImage
from master_calib_creation.creation import gen_oned_spec, gen_echellogram, gen_echellogram_fit_wvlsol,\
    gen_identified_lines, gen_oned_wavemap, json_dict_from_file, find_nearest
from master_calib_creation.image import file_overlay


def gen_order_map(array_shape, order_height, output_filename='deveny_order_map.fits'):
    output_map = np.zeros(array_shape)
    order_top = int((array_shape[0]+order_height)/2)
    order_bottom = int((array_shape[0]-order_height)/2)
    output_map[order_top:order_bottom, :] = 1
    ArrayImage(output_map).save(output_filename, 0)


def gen_rough_wavemap(array_shape, start_wvl, end_wvl, output_filename='deveny_wavemap.fits'):
    output_map = np.zeros(array_shape)
    linear_wvl = np.linspace(start_wvl, end_wvl, array_shape[1])
    output_map[:] = linear_wvl
    ArrayImage(output_map).save(output_filename, 0)


def gen_arc_lines(start_wvl=2000, end_wvl=10000, output_filename='CdArNeHg_lines.dat', elements='Hg Ne Ar Cd'):
    line_table = Nist.query(start_wvl*u.AA, end_wvl*u.AA, elements)
    wvls = line_table['Observed']
    intensities = line_table['Rel.'].str.extract('(\d+)', expand=False)
    array = np.asarray((wvls, intensities)).transpose()
    np.savetxt(output_filename, array)


def gen_arc_line_alignment_image(wave_map_file, arc_line_file, output_filename, start_wvl=0, end_wvl=100000000):
    wavemap = ExistingImage(wave_map_file).image
    output_array = np.zeros(wavemap.shape)
    wavelengths = np.unique(wavemap)
    arc_lines = np.loadtxt(arc_line_file)
    arc_lines_trunc = arc_lines[np.logical_and(arc_lines[:, 0] >= start_wvl, arc_lines[:, 0] <= end_wvl)]
    for line, intensity in arc_lines_trunc:
        value, index = find_nearest(wavelengths, line)
        output_array[output_array == value] = intensity
    ArrayImage(output_array, 0).save(output_filename, 0)


def gen_lazy_bad_pix_map(shape, output_filename):
    array = np.zeros(shape)
    ArrayImage(array).save(output_filename, 0)


if __name__ == '__main__':
    arc_spectrum = '../master_calib/deveny/20210506.0014.fits'
    arc_spectrum_oned = '../master_calib/deveny/CdArNeHg_onedspec.json'
    order_map_file = '../master_calib/deveny/deveny_order_map.fits'
    wavemap_file = '../master_calib/deveny/deveny_wavemap.fits'
    wavemap_oned = '../master_calib/deveny/deveny_wavemap_oned.json'
    lines_dat = '../master_calib/deveny/CdArNeHg_lines.dat'
    identified_lines = '../master_calib/deveny/CdArNeHg_identified_lines.json'
    alignment_arc_file = '../master_calib/deveny/CdArNeHg_alignment.fits'
    bad_pix_map = '../master_calib/deveny/deveny_bad_pix_map.fits'
    deveny_start_wvl = 3580
    deveny_end_wvl = 8020

    arc_spectrum_image = ExistingImage(arc_spectrum, fits_image_hdu=0).image
    gen_order_map(arc_spectrum_image.shape, 40, order_map_file)
    gen_rough_wavemap(arc_spectrum_image.shape, deveny_start_wvl, deveny_end_wvl, wavemap_file)
    gen_lazy_bad_pix_map(arc_spectrum_image.shape, bad_pix_map)
    gen_arc_lines(output_filename=lines_dat)
    gen_arc_line_alignment_image(wavemap_file, lines_dat, alignment_arc_file, deveny_start_wvl, deveny_end_wvl)
    file_overlay(alignment_arc_file, arc_spectrum)
    # gen_oned_spec(order_map_file, arc_spectrum, arc_spectrum_oned, 0)
    # gen_oned_wavemap(order_map_file, wavemap_file, wavemap_oned, 0)
    # gen_identified_lines(arc_spectrum_oned, wavemap_oned, lines_dat, identified_lines)
