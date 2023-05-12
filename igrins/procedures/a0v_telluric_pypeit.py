import os

from astropy.coordinates import SkyCoord
import numpy as np

from pypeit.core import flux_calib
from pypeit.core import telluric
#from pypeit.spectrographs.util import load_spectrograph

def flatten_a0v(obsset, fill_nan=None):

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)

    wvl_solutions = helper.get("wvl_solutions")  # extractor.wvl_solutionsw
    domain_list = helper.get("domain_list")

    norders = len(wvl_solutions)

    wvl = []
    s_model = []
    t_model = []
    mask_array = []
    reverse = False
    for i in range(norders):
        wave = wvl_solutions[i]*1e4

        wave = wave[domain_list[i][0]:domain_list[i][1]+1]

        #delwave = np.append(delwave, wave[-1])
        #wave_grid_mid = wave + delwave/2
        #flux = obsset.load_fits_sci_hdu("SPEC_FITS").data[0]
        #var = obsset.load_fits_sci_hdu("VARIANCE_FITS").data[0]
        flux = obsset.load_fits_sci_hdu("SPEC_FITS").data[i]
        var = obsset.load_fits_sci_hdu("VARIANCE_FITS").data[i]
        ivar = 1.0/var
        mask = np.ones_like(flux, dtype=bool)
    
        delwave = wave[1:] - wave[:-1]
        if delwave[0] < 0:
            reverse = True
            wave = wave[::-1]
            flux = flux[::-1]
            ivar = ivar[::-1]
            mask = mask[::-1]

        #Finite Test
        idx = np.isfinite(flux) & np.isfinite(ivar) & np.isfinite(wave) & np.isfinite(mask)
        wave = wave[idx]
        flux = flux[idx]
        ivar = ivar[idx]
        mask = mask[idx]

        #telgridfile = obsset.rs.load_ref_data("TELGRIDFILE_DEVENY")
        telgridfile = obsset.rs.query_ref_value("TELGRIDFILE")
       
        #Should only be really looking at 'A0' stars
        star_type = 'A0' #only really using 'A0' stars
        #star_mag = obsset.recipe_entry['star_mag']
        star_mag = 8.05

        #Get star_ra, dec from FITS file
        hdu = obsset.get_hdus()[0]

        if obsset.expt.lower() == 'deveny':
            print("IN DEVENY")
            c2 = SkyCoord(hdu.header['OBSRA'], hdu.header['OBSDEC'], unit='deg')
            star_ra = c2.ra.deg
            star_dec = c2.dec.deg
        elif obsset.expt.lower() == 'rimas':
            print("ELSE")
            #Alpha Lyr/Vega
            star_ra = 18.00/24.0*360.0 + 36.0/60.0
            star_dec = 38.78

        #Some default values for Pypeit telluric correction
        polyorder = 3
        func = 'legendre'
        model = 'exp'
        only_orders = None
        mask_abs_lines = True
        delta_coeff_bounds = (-20.0, 20.0)
        minmax_coeff_bounds = (-5.0, 5.0)
        maxiter = 2
        debug = debug_init = False
        sn_clip = 30.0
        tol = 1e-3
        popsize = 30
        recombination = 0.7
        polish = True
        disp = False

        #airmass = hdu.header['AIRMASS']
        #exptime = hdu.header['EXPTIME']
        exptime = obsset.recipe_entry['exptime']
        airmass = 1.23

        #star_ra = meta_spec['core']['RA'] if star_ra is None else star_ra
        #star_dec = meta_spec['core']['DEC'] if star_dec is None else star_dec
        std_dict = flux_calib.get_standard_spectrum(star_type=star_type, star_mag=star_mag, ra=star_ra,
                                                    dec=star_dec)

        if flux.ndim == 2:
            norders = flux.shape[1]
        else:
            norders = 1
    
        # Create the polyorder_vec
        if np.size(polyorder) > 1:
            if np.size(polyorder) != norders:
                msgs.error('polyorder must have either have norder elements or be a scalar')
            polyorder_vec = np.array(polyorder)
        else:
            polyorder_vec = np.full(norders, polyorder)

        # Initalize the object parameters
        obj_params = dict(std_dict=std_dict, airmass=airmass,
                          delta_coeff_bounds=delta_coeff_bounds,
                          minmax_coeff_bounds=minmax_coeff_bounds, polyorder_vec=polyorder_vec,
                          exptime=exptime, func=func, model=model, sigrej=3.0,
                          std_ra=std_dict['std_ra'], std_dec=std_dict['std_dec'],
                          std_name=std_dict['name'], std_cal=std_dict['cal_file'],
                          output_meta_keys=('airmass', 'polyorder_vec', 'exptime', 'func', 'std_ra',
                                        'std_dec', 'std_cal'),
                          debug=debug_init)
    
        # Optionally, mask prominent stellar absorption features
        if mask_abs_lines:
            inmask = telluric.mask_star_lines(wave)
            mask_tot = inmask & mask
        else:
            mask_tot = mask
    
        # parameters lowered for testing
        TelObj = telluric.Telluric(wave, flux, ivar, mask_tot, telgridfile, obj_params, telluric.init_star_model,
                                   telluric.eval_star_model,  sn_clip=sn_clip, tol=tol, popsize=popsize,
                                   recombination=recombination, polish=polish, disp=disp, debug=debug,
                                   airmass_guess=airmass)
        TelObj.run(only_orders=only_orders)
    
        # Apply the telluric correction
        telluric2 = TelObj.model['TELLURIC'][0,:]
        star_model = TelObj.model['OBJ_MODEL'][0,:]
  
        import matplotlib.pyplot as plt
        plt.figure("CORRECTIONS")
        plt.plot(wave, telluric2, label='Telluric')
        plt.plot(wave, star_model/np.max(star_model), label='Star Model Normalized')
        plt.legend(loc=0, prop={'size': 12})
        plt.show()

        if reverse:
            wave = wave[::-1]
            telluric2 = telluric2[::-1]
            star_model = star_model[::-1]
            mask = mask[::-1]
            reverse = False

        wvl.append(wave)
        t_model.append(telluric2)
        s_model.append(star_model)
        mask_array.append(mask)
        # Plot the telluric corrected and rescaled spectrum
        #flux_corr = flux*utils.inverse(telluric)
        #ivar_corr = (telluric > 0.0) * ivar * telluric * telluric
        #mask_corr = (telluric > 0.0) * mask
        #sig_corr = np.sqrt(utils.inverse(ivar_corr))

    data_list = [("wavelength", np.array(wvl)),
                ("star_model", np.array(star_model)),
                ("mask", np.array(mask_array)),
                ("telluric_model", np.array(t_model)),
                ("domain_list", np.array(domain_list)),
                ]
   
    #if fill_nan is not None:
    #    flattened_s = data_list[0][1]
    #    flattened_s[~np.isfinite(flattened_s)] = fill_nan

    store_a0v_results(obsset, data_list)

def store_a0v_results(obsset, a0v_flattened_data):

    from .target_spec import get_wvl_header_data
    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    image_list = []
    #image_list.append(([("EXTNAME", "SPEC_FLATTENED")],
    #                   convert_data(a0v_flattened_data[0][1])))

    for ext_name, data in a0v_flattened_data[0:]:
        _ = ([("EXTNAME", ext_name.upper())], convert_data(data))
        image_list.append(_)

    hdul = obsset.get_hdul_to_write(*image_list)
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    hdul[0].verify(option="silentfix")

    obsset.store("spec_fits_flattened", hdul)

def get_a0v_flattened():
    pass

def star_telluric(spec1dfile, telgridfile, telloutfile, outfile, star_type=None, star_mag=None,
                  star_ra=None, star_dec=None, func='legendre', model='exp', polyorder=5,
                  mask_abs_lines=True, delta_coeff_bounds=(-20.0, 20.0),
                  minmax_coeff_bounds=(-5.0, 5.0), only_orders=None, sn_clip=30.0, maxiter=3,
                  tol=1e-3, popsize=30, recombination=0.7, polish=True, disp=False,
                  debug_init=False, debug=False, show=False):
    """
    This needs a doc string.

    USE the one from qso_telluric() as a starting point

    Returns
    -------
    TelObj : :class:`Telluric`
        Object with the telluric modeling results
    """

    # Turn on disp for the differential_evolution if debug mode is turned on.
    if debug:
        disp = True

    # Read in the data
    wave, wave_grid_mid, flux, ivar, mask, meta_spec, header = general_spec_reader(spec1dfile, ret_flam=False)
    # Read in standard star dictionary and interpolate onto regular telluric wave_grid
    star_ra = meta_spec['core']['RA'] if star_ra is None else star_ra
    star_dec = meta_spec['core']['DEC'] if star_dec is None else star_dec
    std_dict = flux_calib.get_standard_spectrum(star_type=star_type, star_mag=star_mag, ra=star_ra,
                                                dec=star_dec)

    if flux.ndim == 2:
        norders = flux.shape[1]
    else:
        norders = 1

    # Create the polyorder_vec
    if np.size(polyorder) > 1:
        if np.size(polyorder) != norders:
            msgs.error('polyorder must have either have norder elements or be a scalar')
        polyorder_vec = np.array(polyorder)
    else:
        polyorder_vec = np.full(norders, polyorder)

    # Initalize the object parameters
    obj_params = dict(std_dict=std_dict, airmass=meta_spec['core']['AIRMASS'],
                      delta_coeff_bounds=delta_coeff_bounds,
                      minmax_coeff_bounds=minmax_coeff_bounds, polyorder_vec=polyorder_vec,
                      exptime=meta_spec['core']['EXPTIME'], func=func, model=model, sigrej=3.0,
                      std_ra=std_dict['std_ra'], std_dec=std_dict['std_dec'],
                      std_name=std_dict['name'], std_cal=std_dict['cal_file'],
                      output_meta_keys=('airmass', 'polyorder_vec', 'exptime', 'func', 'std_ra',
                                        'std_dec', 'std_cal'),
                      debug=debug_init)

    # Optionally, mask prominent stellar absorption features
    if mask_abs_lines:
        inmask = mask_star_lines(wave)
        mask_tot = inmask & mask
    else:
        mask_tot = mask

    # parameters lowered for testing
    TelObj = telluric.Telluric(wave, flux, ivar, mask_tot, telgridfile, obj_params, init_star_model,
                               eval_star_model,  sn_clip=sn_clip, tol=tol, popsize=popsize,
                               recombination=recombination, polish=polish, disp=disp, debug=debug)
    TelObj.run(only_orders=only_orders)
    TelObj.to_file(telloutfile, overwrite=True)

    # Apply the telluric correction
    telluric = TelObj.model['TELLURIC'][0,:]
    star_model = TelObj.model['OBJ_MODEL'][0,:]
    # Plot the telluric corrected and rescaled spectrum
    flux_corr = flux*utils.inverse(telluric)
    ivar_corr = (telluric > 0.0) * ivar * telluric * telluric
    mask_corr = (telluric > 0.0) * mask
    sig_corr = np.sqrt(utils.inverse(ivar_corr))

    if show:
        # TODO: This should get moved into a Telluric.show() method.
        # Median filter
        fig = plt.figure(figsize=(12, 8))
        plt.plot(wave, flux_corr*mask_corr, drawstyle='steps-mid', color='k',
                 label='corrected data', alpha=0.7, zorder=5)
        plt.plot(wave, flux*mask_corr, drawstyle='steps-mid', color='0.7',
                 label='uncorrected data', alpha=0.7, zorder=3)
        plt.plot(wave, sig_corr*mask_corr, drawstyle='steps-mid', color='r', label='noise',
                 alpha=0.3, zorder=1)
        plt.plot(wave, star_model, color='cornflowerblue', linewidth=1.0,
                 label='poly scaled star model', zorder=7, alpha=0.7)
        plt.plot(std_dict['wave'].value, std_dict['flux'].value, color='green', linewidth=1.0,
                 label='original star model', zorder=8, alpha=0.7)
        plt.plot(wave, star_model.max()*0.9*telluric, color='magenta', drawstyle='steps-mid',
                 label='telluric', alpha=0.4)
        plt.ylim(-np.median(sig_corr[mask_corr]).max(), 1.5*star_model.max())
        plt.xlim(wave[wave > 1.0].min(), wave[wave > 1.0].max())
        plt.legend()
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.show()

    # save the telluric corrected spectrum
    save_coadd1d_tofits(outfile, wave, flux_corr, ivar_corr, mask_corr, wave_grid_mid=wave_grid_mid,
                        spectrograph=header['PYP_SPEC'], telluric=telluric,
                        obj_model=star_model, header=header, ex_value='OPT', overwrite=True)

    return TelObj
