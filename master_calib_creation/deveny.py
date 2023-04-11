import re
import os
from shutil import copy2

from astroquery.nist import Nist
from astropy import units as u
from astropy.io import fits
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


def gen_arc_line_alignment_image(wave_map_file, arc_line_file, output_filename, start_wvl=0., end_wvl=100000000.):
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
    specs = np.asarray([np.asarray(json_dict_from_file(f)['specs'][0]) for f in list_oned_spec_files])
    fnames = [os.path.basename(f) for f in list_oned_spec_files]
    med_spec = np.asarray([np.median(spec) for spec in specs])
    specs = [spec - med for spec, med in zip(specs, med_spec)]
    max_spec = np.asarray([np.max(spec) for spec in specs])
    norm_spec = np.min(max_spec) / max_spec
    normed_specs = [spec*n for spec,n in zip(specs, norm_spec)]
    for spec, f in zip(normed_specs, fnames):
        plt.plot(spec, label=f)
        print(spec.shape)
    plt.show()


if __name__ == '__main__':
    run_gen_wavemap = False
    run_arc_alignment = False
    run_oh_alignment = False
    run_gen_arc_id_lines = False
    run_gen_oh_id_lines = False
    run_gen_echellogram = True

    # begin generating reduced 2D spectra
    obsdir = '/Users/jdurbak/Downloads/20221210a'
    obsdate = '20221210'
    arc_elements = 'Ar I, Ne I'
    p_init = None
    arc_elements_fname = ''.join(arc_elements.split(' '))
    arc_elements_fname = ''.join(arc_elements_fname.split(','))
    file_format = os.path.join(obsdir, obsdate + '.{:04d}.fits')
    bias_range = range(1, 10+1)
    arcs_range = range(18, 27+1)
    flats_range = range(28, 32+1)
    sky_range = [47, 48]
    arcs = '../master_calib/deveny/arcs.{}.fits'.format(arc_elements_fname)
    bias = '../master_calib/deveny/bias.fits'
    flat = '../master_calib/deveny/flat.fits'
    sky = '../master_calib/deveny/sky.fits'
    cal_names = [bias, arcs, flat, sky]
    cal_ranges = [bias_range, arcs_range, flats_range, sky_range]
    for cal, cal_range in zip(cal_names, cal_ranges):
        if not os.path.isfile(cal):
            cal_frame = np.median(np.asarray([fits.getdata(file_format.format(i)) for i in cal_range]), axis=0)
            if cal != bias:
                cal_frame = cal_frame - ExistingImage(bias).image
            ArrayImage(cal_frame).save(cal)

    # generate maps
    map_shape = ExistingImage(arcs).image.shape
    order_map_file = '../master_calib/deveny/deveny_order_map.fits'
    if not os.path.isfile(order_map_file):
        gen_order_map(map_shape, 25, order_map_file)
    bad_pix_map = '../master_calib/deveny/deveny_bad_pix_map.fits'
    if not os.path.isfile(bad_pix_map):
        gen_lazy_bad_pix_map(map_shape, bad_pix_map)

    # wavemap settings
    wavemap_file = '../master_calib/deveny/deveny_wavemap.fits'
    wavemap_microns = wavemap_file.replace('.fits', '.microns.fits')
    wavemap_oned = '../master_calib/deveny/deveny_wavemap_oned.json'
    wavemap_microns_oned = '../master_calib/deveny/deveny_wavemap_microns_oned.json'
    pixel_wvl_shift = -4
    deveny_start_wvl = 9032.258527830243 + pixel_wvl_shift - 4
    deveny_end_wvl = 10826.582530520567 + pixel_wvl_shift + 7
    # pixel_wvl_shift = -15
    # deveny_start_wvl = 9029.75248872035 + pixel_wvl_shift
    # deveny_end_wvl = 10824.076491410671 + pixel_wvl_shift +10
    if run_gen_wavemap:
        gen_rough_wavemap(map_shape, deveny_start_wvl, deveny_end_wvl, wavemap_file)
        ArrayImage(ExistingImage(wavemap_file).image / 10000).save(wavemap_microns, hdu=0)
        gen_oned_spec(order_map_file, wavemap_file, wavemap_oned)
        gen_oned_spec(order_map_file, wavemap_microns, wavemap_microns_oned)

    # begin generating reduced 1D spectra
    bias_oned = '../master_calib/deveny/bias_oned.json'
    arcs_oned = '../master_calib/deveny/arcs_oned.json'
    flat_oned = '../master_calib/deveny/flat_oned.json'
    sky_oned = '../master_calib/deveny/sky_oned.json'
    oned_names = [bias_oned, arcs_oned, flat_oned, sky_oned]
    for cal, oned in zip(cal_names, oned_names):
        if not os.path.isfile(oned):
            gen_oned_spec(order_map_file, cal, oned)

    # generate support files
    lines_dat = '../master_calib/deveny/{}_lines.dat'.format(arc_elements_fname)
    if not os.path.isfile(lines_dat):
        gen_arc_lines(output_filename=lines_dat, elements=arc_elements, start_wvl=8000, end_wvl=10000)

    ohlines_dat = '../master_calib/deveny/ohlines.dat'
    if not os.path.isfile(ohlines_dat):
        copy2('../master_calib/igrins/ohlines.dat', ohlines_dat)

    # spectrum alignment
    if run_arc_alignment:
        alignment_arc_file = '../master_calib/deveny/{}_alignment.fits'.format(arc_elements_fname)
        alignment_arc_file_reshaped = alignment_arc_file.replace('.fits', '_reshaped.fits')
        alignment_arc_reshaped_oned = alignment_arc_file.replace('.fits', '_oned.json')
        gen_arc_line_alignment_image(wavemap_file, lines_dat, alignment_arc_file, deveny_start_wvl, deveny_end_wvl)
        alignment_arc = ExistingImage(alignment_arc_file, fits_image_hdu=0)
        alignment_arc.scale_and_translate(0, 0, 1.0, 1.0)
        alignment_arc.save(alignment_arc_file_reshaped, 0)
        # file_overlay(alignment_arc_file_reshaped, arcs)
        gen_oned_spec(order_map_file, alignment_arc_file_reshaped, alignment_arc_reshaped_oned)
        plot_oned_spec((arcs_oned, alignment_arc_reshaped_oned))

    if run_oh_alignment:
        alignment_oh_file = '../master_calib/deveny/{}_alignment.fits'.format('oh')
        alignment_oh_file_reshaped = alignment_oh_file.replace('.fits', '_reshaped.fits')
        alignment_oh_reshaped_oned = alignment_oh_file.replace('.fits', '_oned.json')
        gen_arc_line_alignment_image(wavemap_file, ohlines_dat, alignment_oh_file, deveny_start_wvl, deveny_end_wvl)
        alignment_arc = ExistingImage(alignment_oh_file, fits_image_hdu=0)
        alignment_arc.scale_and_translate(0, 0, 1.0, 1.0)
        alignment_arc.save(alignment_oh_file_reshaped, 0)
        # file_overlay(alignment_oh_file_reshaped, sky)
        gen_oned_spec(order_map_file, alignment_oh_file_reshaped, alignment_oh_reshaped_oned)
        plot_oned_spec((
            sky_oned,
            alignment_oh_reshaped_oned
        ))

    # generate identified lines and ref_indices
    oh_id_lines = '../master_calib/deveny/oh_id_lines.json'
    oh_ref_indices_file = '../master_calib/deveny/oh_ref_indices.json'
    arc_id_lines = '../master_calib/deveny/{}_id_lines.json'.format(arc_elements_fname)
    arc_ref_indices_file = '../master_calib/deveny/{}_ref_indices.json'.format(arc_elements_fname)
    if run_gen_arc_id_lines:
        gen_identified_lines(
            arcs_oned, wavemap_microns_oned, lines_dat, arc_id_lines, arc_ref_indices_file, 'deveny',
            p_init_pickle=p_init,
            plt_peak=True, plt_wvl=False, plt_pix=True, manual_filter_peak=True,
            domain_starting_pixel=0, domain_ending_pixel=map_shape[1],
            sigma_filter=False, peak_finder_sigma=1
        )
    if run_gen_oh_id_lines:
        gen_identified_lines(
            sky_oned, wavemap_microns_oned, ohlines_dat, oh_id_lines, oh_ref_indices_file, 'deveny',
            p_init_pickle=p_init,
            plt_peak=True, plt_wvl=False, plt_pix=True, manual_filter_peak=True,
            domain_starting_pixel=0, domain_ending_pixel=map_shape[1],
            sigma_filter=False, peak_finder_sigma=1
        )

    # generate echellogram
    echellogram_file = '../master_calib/deveny/deveny_echellogram.json'
    echellogram_file_wvl_sol = '../master_calib/deveny/deveny_echellogram_wvl_sol.json'
    fit_pickle_file = '../master_calib/deveny/deveny_fit.p'
    if run_gen_echellogram:
        gen_echellogram(order_map_file, wavemap_oned, echellogram_file)
        gen_echellogram_fit_wvlsol(
            echellogram_file, [arc_id_lines], [arc_ref_indices_file],
            echellogram_file_wvl_sol,
            map_shape[1], pixel_degree=3, order_degree=0, pickle_output_file=fit_pickle_file, band='deveny', sigma=2
        )
