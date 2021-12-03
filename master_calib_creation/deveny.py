import re
import os

from astroquery.nist import Nist
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

from master_calib_creation.image import ExistingImage, ArrayImage
from master_calib_creation.creation import gen_oned_spec, gen_echellogram, gen_echellogram_fit_wvlsol,\
    gen_identified_lines, gen_oned_wavemap, json_dict_from_file, find_nearest, gen_ref_indices
from master_calib_creation.image import file_overlay


def gen_order_map(array_shape, order_height, output_filename='deveny_order_map.fits'):
    output_map = np.zeros(array_shape)
    order_top = int((array_shape[0]+order_height)/2)
    order_bottom = int((array_shape[0]-order_height)/2)
    output_map[order_bottom:order_top, :] = 1
    ArrayImage(output_map).save(output_filename, 0)


def gen_rough_wavemap(array_shape, start_wvl, end_wvl, output_filename='deveny_wavemap.fits'):
    output_map = np.zeros(array_shape)
    linear_wvl = np.linspace(end_wvl, start_wvl, array_shape[1])
    output_map[:] = linear_wvl
    ArrayImage(output_map).save(output_filename, 0)


def gen_arc_lines(start_wvl=2000, end_wvl=10000, output_filename='CdArNeHg_lines.dat', elements='Hg Ne Ar Cd'):
    line_table = Nist.query(start_wvl*u.AA, end_wvl*u.AA, elements)
    wvls = line_table['Observed']
    # intensities = line_table['Rel.'].str.extract('(\d+)', expand=False)
    intensities = line_table['Rel.']
    try:
        intensities_mask = intensities.mask
    except AttributeError:
        intensities_mask = np.zeros(intensities.shape).astype(np.bool)
    try:
        wvls_mask = wvls.mask
    except AttributeError:
        wvls_mask = np.zeros(wvls.shape).astype(np.bool)

    not_masked = np.logical_not(np.logical_or(wvls_mask, intensities_mask))

    try:
        wvls_unmasked = wvls[not_masked].filled()
    except AttributeError:
        wvls_unmasked = wvls[not_masked]

    try:
        intensities_unmasked = intensities[not_masked].filled()
    except AttributeError:
        intensities_unmasked = intensities[not_masked]

    output_list = []
    for wvl, intensity in zip(wvls_unmasked, intensities_unmasked):
        try:
            intensity_int = np.int(re.findall(r'\d+', intensity)[0])
            wvl_float = np.float64(wvl)
            output_list.append((wvl_float, intensity_int))
        except (TypeError, ValueError, IndexError):
            pass

    np.savetxt(output_filename, np.asarray(output_list))


def gen_arc_line_alignment_image(wave_map_file, arc_line_file, output_filename, start_wvl=0, end_wvl=100000000):
    wavemap = ExistingImage(wave_map_file).image
    output_array = np.ones(wavemap.shape)
    wavelengths = np.unique(wavemap)
    arc_lines = np.loadtxt(arc_line_file)
    arc_lines_trunc = arc_lines[np.logical_and(arc_lines[:, 0] >= start_wvl, arc_lines[:, 0] <= end_wvl)]
    for line, intensity in arc_lines_trunc:
        value, index = find_nearest(wavelengths, line)
        output_array[wavemap == value] = intensity
        # output_array[wavemap == value] = 1000
    ArrayImage(output_array).save(output_filename, 0)


def gen_lazy_bad_pix_map(shape, output_filename):
    array = np.zeros(shape)
    ArrayImage(array).save(output_filename, 0)


def plot_oned_spec(list_oned_spec_files):
    specs = np.asarray([json_dict_from_file(f)['specs'][0] for f in list_oned_spec_files])
    fnames = [os.path.basename(f) for f in list_oned_spec_files]
    max_spec = np.asarray([np.max(spec) for spec in specs])
    norm_spec = np.min(max_spec) / max_spec
    normed_specs = [spec*n for spec,n in zip(specs, norm_spec)]
    for spec, f in zip(normed_specs, fnames):
        plt.plot(spec, label=f)
    plt.show()


if __name__ == '__main__':
    arc_spectrum = '../master_calib/deveny/20210506.0014.fits'
    # arc_spectrum = '../master_calib/deveny/20190108.0043.fits'
    bias = '../master_calib/deveny/bias.fits'
    arc_elements = 'Cd Ar Ne Hg'
    # arc_elements = 'Hg'
    unbiased_arc_spectrum = ExistingImage(arc_spectrum).image - ExistingImage(bias).image
    arc_spectrum = '../master_calib/deveny/{}.fits'.format(arc_elements)
    ArrayImage(unbiased_arc_spectrum).save(arc_spectrum, hdu=0)
    arc_elements_fname = ''.join(arc_elements.split(' '))
    arc_spectrum_oned = '../master_calib/deveny/{}_onedspec.json'.format(arc_elements_fname)

    sky_plus = '../master_calib/deveny/20210506.0037.fits'
    unbiased_sky_spectrum = ExistingImage(sky_plus).image - ExistingImage(bias).image
    sky_spectrum = '../master_calib/deveny/sky.fits'
    ArrayImage(unbiased_sky_spectrum).save(sky_spectrum, hdu=0)
    sky_spectrum_oned = '../master_calib/deveny/sky_onedspec.json'

    order_map_file = '../master_calib/deveny/deveny_order_map.fits'
    wavemap_file = '../master_calib/deveny/deveny_wavemap.fits'
    wavemap_microns = wavemap_file.replace('.fits', '_microns.fits')
    wavemap_oned = '../master_calib/deveny/deveny_wavemap_oned.json'
    # lines_dat = '../master_calib/deveny/{}_lines.dat'.format(arc_elements_fname)
    lines_dat = '../master_calib/deveny/{}_lines.dat'.format('deveny')
    ohlines_dat = '../master_calib/deveny/ohlines.dat'
    oh_identified_lines = '../master_calib/deveny/{}_identified_lines.json'.format('OH')
    identified_lines = '../master_calib/deveny/{}_identified_lines.json'.format(arc_elements_fname)
    identified_lines_update = identified_lines.replace('.json', '_update.json')
    oh_identified_lines_update = oh_identified_lines.replace('.json', '_update.json')
    alignment_arc_file = '../master_calib/deveny/{}_alignment.fits'.format(arc_elements_fname)
    alignment_arc_file_reshaped = alignment_arc_file.replace('.fits', '_reshaped.fits')
    alignment_arc_reshaped_oned = alignment_arc_file.replace('.fits', '_oned.json')
    bad_pix_map = '../master_calib/deveny/deveny_bad_pix_map.fits'
    echellogram_file = '../master_calib/deveny/deveny_echellogram.json'
    echellogram_file_wvl_sol = '../master_calib/deveny/deveny_echellogram_wvl_sol.json'
    ref_indices_file = '../master_calib/deveny/deveny_ref_indices.json'
    oh_ref_indices_file = '../master_calib/deveny/oh_ref_indices.json'
    fit_pickle_file = '../master_calib/deveny/deveny_fit.p'
    deveny_start_wvl = 3424.5864
    deveny_end_wvl = 8098.6615

    arc_spectrum_image = ExistingImage(arc_spectrum, fits_image_hdu=0).image
    # gen_order_map(arc_spectrum_image.shape, 25, order_map_file)
    # gen_rough_wavemap(arc_spectrum_image.shape, deveny_start_wvl, deveny_end_wvl, wavemap_file)
    # ArrayImage(ExistingImage(wavemap_file).image/10000).save(wavemap_microns, hdu=0)
    # gen_lazy_bad_pix_map(arc_spectrum_image.shape, bad_pix_map)
    # gen_arc_lines(output_filename=lines_dat, elements=arc_elements)
    # gen_arc_line_alignment_image(wavemap_file, lines_dat, alignment_arc_file, deveny_start_wvl, deveny_end_wvl)
    # alignment_arc = ExistingImage(alignment_arc_file, fits_image_hdu=0)
    # alignment_arc.scale_and_translate(0,0,1.0,1.0)
    # alignment_arc.save(alignment_arc_file_reshaped, 0)
    # file_overlay(alignment_arc_file_reshaped, arc_spectrum)
    # gen_oned_spec(order_map_file, arc_spectrum, arc_spectrum_oned, 0)
    ##gen_oned_spec(order_map_file, sky_spectrum, sky_spectrum_oned, 0)
    # gen_oned_spec(order_map_file, alignment_arc_file_reshaped, alignment_arc_reshaped_oned, 0)
    # gen_oned_wavemap(order_map_file, wavemap_microns, wavemap_oned, 0)
    # gen_identified_lines(arc_spectrum_oned, wavemap_oned, lines_dat, identified_lines)
    # gen_identified_lines(sky_spectrum_oned, wavemap_oned, ohlines_dat, oh_identified_lines)
    gen_ref_indices(identified_lines, lines_dat, 'deveny', identified_lines_update, ref_indices_file)
    gen_ref_indices(oh_identified_lines, ohlines_dat, 'deveny', oh_identified_lines_update, oh_ref_indices_file)
    # gen_echellogram(order_map_file, wavemap_oned, echellogram_file)
    # gen_echellogram_fit_wvlsol(
    #     echellogram_file, identified_lines_update, ref_indices_file, echellogram_file_wvl_sol,
    #     arc_spectrum_image.shape[1], pixel_degree=4, order_degree=0, pickle_output_file=fit_pickle_file, band='deveny'
    # )
    # plot_oned_spec((sky_spectrum_oned, alignment_arc_reshaped_oned))
