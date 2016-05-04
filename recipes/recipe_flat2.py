from libs.recipe_helper import RecipeHelper
import numpy as np
import scipy.ndimage as ni

import astropy.io.fits as pyfits
from astropy.io.fits import Card

from libs.process_flat import FlatOff, FlatOn

def get_data_list(helper, band, obsids):
    _ = helper.get_base_info(band, obsids)
    flat_off_filenames = _[0]

    flat_offs_hdu_list = [pyfits.open(fn_)[0] for fn_
                          in flat_off_filenames]

    flat_offs = [hdu.data for hdu in flat_offs_hdu_list]

    return flat_offs


def get_combined_image(data_list, destripe=True):
    # destripe=True):


    from libs.stsci_helper import stsci_median
    flat_off = stsci_median(data_list)

    flat_off_cards = []

    if destripe:
        from libs.destriper import destriper
        flat_off = destriper.get_destriped(flat_off)

        flat_off_cards.append(Card("HISTORY",
                                   "IGR: image destriped."))

    return flat_off, flat_off_cards


def combine_flat_off(helper, band, obsids, destripe=True):
    # destripe=True):

    data_list = get_data_list(helper, band, obsids)

    flat_off, flat_off_cards = get_combined_image(data_list,
                                                  destripe=True)

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    caldb.store_image((band, master_obsid),
                      item_type="flat_off", data=flat_off,
                      card_list=flat_off_cards)


def make_hotpix_mask(helper, band, obsids_off,
                     sigma_clip1=100, sigma_clip2=10,
                     medfilter_size=None):

    caldb = helper.get_caldb()
    master_obsid = obsids_off[0]

    flat_off_hdu = caldb.load_item_from((band, master_obsid), "flat_off")[0]
    flat_off = flat_off_hdu.data

    import libs.badpixel as bp
    hotpix_mask = bp.badpixel_mask(flat_off,
                                   sigma_clip1=sigma_clip1,
                                   sigma_clip2=sigma_clip2,
                                   medfilter_size=medfilter_size)

    bg_std = flat_off[~hotpix_mask].std()

    flat_off_cards = [Card("BG_STD", bg_std, "IGR: stddev of combined flat")]

    caldb = helper.get_caldb()

    caldb.store_image((band, master_obsid),
                      item_type="hotpix_mask", data=hotpix_mask)

   # save fits with updated header
    caldb.store_image((band, master_obsid),
                      item_type="flat_off", data=flat_off,
                      header=flat_off_hdu.header,
                      card_list=flat_off_cards)


def process_flat_off(helper, band, obsids_off):

    combine_flat_off(helper, band, obsids_off)

    make_hotpix_mask(helper, band, obsids_off,
                     sigma_clip1=100, sigma_clip2=5)



## flat on

from collections import namedtuple
SimpleHDU = namedtuple('SimpleHDU', ['header', 'data'])

def combine_flat_on(helper, band, obsids):

    caldb = helper.get_caldb()

    data_list = get_data_list(helper, band, obsids)

    flat_on, flat_on_cards = get_combined_image(data_list, destripe=False)

    flat_std = np.std(data_list, axis=0)

    hdu_list = [SimpleHDU(flat_on_cards, flat_on),
                SimpleHDU([], flat_std)]

    master_obsid = obsids[0]
    caldb.store_multi_image((band, master_obsid),
                            item_type="flat_on",
                            hdu_list=hdu_list)

                            # item_type="flat_on", data=flat_on - flat_off,
                            # card_list=flat_on_cards)


def make_deadpix_mask(helper, band, obsids,
                      flat_mask_sigma=5.,
                      deadpix_thresh=0.6,
                      smooth_size=9):

    # def make_flaton_deadpixmap(self, flatoff_product,
    #                            deadpix_mask_old=None,
    #                            flat_mask_sigma=5.,
    #                            deadpix_thresh=0.6,
    #                            smooth_size=9):

    caldb = helper.get_caldb()
    master_obsid = obsids[0]

    from libs.master_calib import load_ref_data
    f = load_ref_data(helper.config, band=band,
                      kind="DEFAULT_DEADPIX_MASK")

    deadpix_mask_old = f[0].data.astype(bool)


    flat_on_off_hdus = caldb.load_item_from((band, master_obsid),
                                            "flat_on")

    flat_on_off = flat_on_off_hdus[0].data
    flat_std = flat_on_off_hdus[1].data

    hotpix_mask = caldb.load_resource_for((band, master_obsid),
                                          "hotpix_mask").data

    if 1:

        # normalize it
        from libs.trace_flat import (get_flat_normalization, get_flat_mask,
                                     get_flat_mask_auto,
                                     estimate_bg_mean_std)
        bg_mean, bg_fwhm = estimate_bg_mean_std(flat_on_off)
        norm_factor = get_flat_normalization(flat_on_off,
                                             bg_fwhm, hotpix_mask)

        flat_normed = flat_on_off / norm_factor
        flat_std_normed = ni.median_filter(flat_std / norm_factor,
                                           size=(3,3))
        bg_fwhm_norm = bg_fwhm/norm_factor

        # mask out bpix
        flat_bpixed = flat_normed.astype("d") # by default, astype
                                              # returns new array.
        flat_bpixed[hotpix_mask] = np.nan

        flat_mask = get_flat_mask_auto(flat_bpixed)
        # flat_mask = get_flat_mask(flat_bpixed, bg_std_norm,
        #                           sigma=flat_mask_sigma)


        # get dead pixel mask
        flat_smoothed = ni.median_filter(flat_normed,
                                         [smooth_size, smooth_size])
        #flat_smoothed[order_map==0] = np.nan
        flat_ratio = flat_normed/flat_smoothed
        flat_std_mask = (flat_smoothed - flat_normed) > 5*flat_std_normed

        refpixel_mask = np.ones(flat_mask.shape, bool)
        # mask out outer boundaries
        refpixel_mask[4:-4,4:-4] = False

        deadpix_mask = (flat_ratio<deadpix_thresh) & flat_std_mask & flat_mask & (~refpixel_mask)

        if deadpix_mask_old is not None:
            deadpix_mask = deadpix_mask | deadpix_mask_old

        flat_bpixed[deadpix_mask] = np.nan

        basename = (band, master_obsid)
        caldb.store_image(basename, "flat_normed", data=flat_normed)
        caldb.store_image(basename, "flat_bpixed", data=flat_bpixed)
        caldb.store_image(basename, "flat_mask", data=flat_mask)
        caldb.store_image(basename, "deadpix_mask", data=deadpix_mask)

        caldb.store_dict(basename, "flaton_json",
                         data=dict(bg_fwhm_norm=bg_fwhm_norm))


# class FlatOn(object):
#     def __init__(self, ondata_list):
#         self.data_list = ondata_list

#     def make_flaton_deadpixmap(self, flatoff_product,
#                                deadpix_mask_old=None,
#                                flat_mask_sigma=5.,
#                                deadpix_thresh=0.6,
#                                smooth_size=9):

#         # load flat off data

#         # flat_off = flatoff_product["flat_off"]
#         # bg_std = flatoff_product["bg_std"]
#         # hotpix_mask = flatoff_product["hotpix_mask"]

#         from storage_descriptions import (FLAT_OFF_DESC,
#                                           FLATOFF_JSON_DESC,
#                                           HOTPIX_MASK_DESC)


#         flat_off = flatoff_product[FLAT_OFF_DESC].data
#         bg_std = flatoff_product[FLATOFF_JSON_DESC]["bg_std"]
#         hotpix_mask = flatoff_product[HOTPIX_MASK_DESC].data

#         flat_on = stsci_median(self.data_list)
#         flat_on_off = flat_on - flat_off

#         # normalize it
#         norm_factor = get_flat_normalization(flat_on_off,
#                                              bg_std, hotpix_mask)

#         flat_normed = flat_on_off / norm_factor
#         flat_std_normed = ni.median_filter(np.std(self.data_list, axis=0) / norm_factor,
#                                            size=(3,3))
#         bg_std_norm = bg_std/norm_factor

#         # mask out bpix
#         flat_bpixed = flat_normed.astype("d") # by default, astype
#                                               # returns new array.
#         flat_bpixed[hotpix_mask] = np.nan

#         flat_mask = get_flat_mask_auto(flat_bpixed)
#         # flat_mask = get_flat_mask(flat_bpixed, bg_std_norm,
#         #                           sigma=flat_mask_sigma)


#         # get dead pixel mask
#         flat_smoothed = ni.median_filter(flat_normed,
#                                          [smooth_size, smooth_size])
#         #flat_smoothed[order_map==0] = np.nan
#         flat_ratio = flat_normed/flat_smoothed
#         flat_std_mask = (flat_smoothed - flat_normed) > 5*flat_std_normed

#         refpixel_mask = np.ones(flat_mask.shape, bool)
#         # mask out outer boundaries
#         refpixel_mask[4:-4,4:-4] = False

#         deadpix_mask = (flat_ratio<deadpix_thresh) & flat_std_mask & flat_mask & (~refpixel_mask)

#         if deadpix_mask_old is not None:
#             deadpix_mask = deadpix_mask | deadpix_mask_old

#         flat_bpixed[deadpix_mask] = np.nan


#         from storage_descriptions import (FLAT_NORMED_DESC,
#                                           FLAT_BPIXED_DESC,
#                                           FLAT_MASK_DESC,
#                                           DEADPIX_MASK_DESC,
#                                           FLATON_JSON_DESC)


#         r = PipelineProducts("flat on products")

#         r.add(FLAT_NORMED_DESC, PipelineImageBase([], flat_normed))
#         r.add(FLAT_BPIXED_DESC, PipelineImageBase([], flat_bpixed))
#         r.add(FLAT_MASK_DESC, PipelineImageBase([], flat_mask))
#         r.add(DEADPIX_MASK_DESC, PipelineImageBase([], deadpix_mask))

#         r.add(FLATON_JSON_DESC,
#               PipelineDict(bg_std_normed=bg_std_norm))

#         return r



from libs.trace_flat import (get_y_derivativemap,
                             identify_horizontal_line,
                             trace_centroids_chevyshev)

def identify_order_boundaries(helper, band, obsids_on):

    # flat_normed=flaton_products["flat_normed"]
    # flat_bpixed=flaton_products["flat_bpixed"]
    # bg_std_normed=flaton_products["bg_std_normed"]
    # flat_mask=flaton_products["flat_mask"]

    # from libs.storage_descriptions import (FLAT_NORMED_DESC,
    #                                        FLAT_BPIXED_DESC,
    #                                        FLAT_MASK_DESC,
    #                                        FLATON_JSON_DESC)


    # flat_normed = flaton_products[FLAT_NORMED_DESC][0].data
    # flat_bpixed = flaton_products[FLAT_BPIXED_DESC][0].data
    # flat_mask = flaton_products[FLAT_MASK_DESC][0].data
    # bg_std_normed = flaton_products[FLATON_JSON_DESC]["bg_std_normed"]

    #deadpix_mask=deadpix_mask)

    
    caldb = helper.get_caldb()
    basename = (band, obsids_on[0])

    flat_normed = caldb.load_image(basename, "flat_normed")
    flat_bpixed = caldb.load_image(basename, "flat_bpixed")
    flat_mask = caldb.load_image(basename, "flat_mask")

    flaton_info = caldb.load_item_from(basename, "flaton_json")
    bg_fwhm_normed = flaton_info["bg_fwhm_norm"]

    from libs.trace_flat import get_y_derivativemap
    flat_deriv_ = get_y_derivativemap(flat_normed, flat_bpixed,
                                      bg_fwhm_normed,
                                      max_sep_order=150, pad=10,
                                      flat_mask=flat_mask)

    flat_deriv, flat_deriv_pos_msk, flat_deriv_neg_msk = \
                flat_deriv_["data"], flat_deriv_["pos_mask"], flat_deriv_["neg_mask"]

    # import astropy.io.fits as pyfits
    # for dn in ["data", "pos_mask", "neg_mask"]:
    #     im = pyfits.PrimaryHDU(data=flat_deriv_[dn].astype("d"))
    #     im.writeto("{}.fits".format(dn), clobber=True)


    hdu_list = [SimpleHDU([], flat_deriv),
                SimpleHDU([], flat_deriv_pos_msk),
                SimpleHDU([], flat_deriv_neg_msk)]

    caldb.store_multi_image(basename, item_type="flat_deriv",
                            hdu_list=hdu_list)


def check_boundary_orders(cent_list, nx=2048):

    c_list = []
    for xc, yc in cent_list:
        p = np.polyfit(xc[~yc.mask], yc.data[~yc.mask], 2)
        c_list.append(np.polyval(p, nx/2.))

    indexes = np.argsort(c_list)

    return [cent_list[i] for i in indexes]


def trace_order_boundaries(helper, band, obsids_on):

    caldb = helper.get_caldb()

    basename = (band, obsids_on[0])
    hdu_list = caldb.load_item_from(basename, "flat_deriv")

    flat_deriv = hdu_list[0].data
    flat_deriv_pos_msk = hdu_list[1].data > 0
    flat_deriv_neg_msk = hdu_list[2].data > 0

    flaton_info = caldb.load_item_from(basename, "flaton_json")
    bg_fwhm_normed = flaton_info["bg_fwhm_norm"]

    ny, nx = flat_deriv.shape
    from libs.trace_flat import identify_horizontal_line

    cent_bottom_list = identify_horizontal_line(flat_deriv,
                                                flat_deriv_pos_msk,
                                                pad=10,
                                                bg_std=bg_fwhm_normed)

    cent_bottom_list = check_boundary_orders(cent_bottom_list, nx=2048)

    cent_up_list = identify_horizontal_line(-flat_deriv,
                                            flat_deriv_neg_msk,
                                            pad=10,
                                            bg_std=bg_fwhm_normed)

    cent_up_list = check_boundary_orders(cent_up_list, nx=2048)

    caldb.store_dict(basename, "flatcentroids_json",
                     dict(bottom_centroids=cent_bottom_list,
                          up_centroids=cent_up_list))


def stitch_up_traces(helper, band, obsids_on):
    # from libs.process_flat import trace_solutions
    # trace_solution_products, trace_solution_products_plot = \
    #                          trace_solutions(trace_products)

    caldb = helper.get_caldb()
    basename = (band, obsids_on[0])

    centroids_dict = caldb.load_item_from(basename, "flatcentroids_json")
    
    bottom_centroids = centroids_dict["bottom_centroids"]
    up_centroids = centroids_dict["up_centroids"]

    from libs.igrins_detector import IGRINSDetector
    nx = IGRINSDetector.nx

    from libs.trace_flat import trace_centroids_chevyshev
    _ = trace_centroids_chevyshev(bottom_centroids,
                                  up_centroids,
                                  domain=[0, nx],
                                  ref_x=nx/2)

    bottom_up_solutions_full, bottom_up_solutions, bottom_up_centroids = _

    assert len(bottom_up_solutions_full) != 0

    from numpy.polynomial import Polynomial

    bottom_up_solutions_as_list = []

    for b, d in bottom_up_solutions_full:

        bb, dd = b.convert(kind=Polynomial), d.convert(kind=Polynomial)
        bb_ = ("poly", bb.coef)
        dd_ = ("poly", dd.coef)
        bottom_up_solutions_as_list.append((bb_, dd_))

    def jsonize_cheb(l):
        return [(repr(l1), l1.coef, l1.domain, l1.window) for l1 in l]

    r = dict(orders=[],
             bottom_up_solutions=bottom_up_solutions_as_list,
             bottom_up_centroids=bottom_up_centroids,
             bottom_up_solutions_qa=[jsonize_cheb(bottom_up_solutions[0]),
                                     jsonize_cheb(bottom_up_solutions[1])])

    caldb.store_dict(basename, "flatcentroid_sol_json", r)

    # from storage_descriptions import FLATCENTROID_SOL_JSON_DESC

    # r = PipelineProducts("order trace solutions")
    # r.add(FLATCENTROID_SOL_JSON_DESC,
    #       PipelineDict(orders=[],
    #                    #bottom_up_centroids=bottom_up_centroids,
    #                    #bottom_up_solutions=bottom_up_solutions,
    #                    bottom_up_solutions=bottom_up_solutions_as_list,
    #                    ))

    # r2 = PipelineProducts("order trace solutions")
    # r2.add(FLATCENTROID_SOL_JSON_DESC,
    #        PipelineDict(#orders=[],
    #                     bottom_up_centroids=bottom_up_centroids,
    #                     bottom_up_solutions=bottom_up_solutions,
    #                     #bottom_up_solutions_full=bottom_up_solutions_as_list,
    #                     ))


def trace_aperture(helper, band, obsids_on):
    # now trace the orders

    identify_order_boundaries(helper, band, obsids_on)
    
    trace_order_boundaries(helper, band, obsids_on)

    stitch_up_traces(helper, band, obsids_on)


def process_flat_on(helper, band, obsids_on):

    combine_flat_on(helper, band, obsids_on)

    make_deadpix_mask(helper, band, obsids_on,
                      flat_mask_sigma=5.,
                      deadpix_thresh=0.6,
                      smooth_size=9)

    trace_aperture(helper, band, obsids_on)


def store_aux_data(helper, band, obsids_on):

    caldb = helper.get_caldb()
    basename = (band, obsids_on[0])

    flatcentroid_info =caldb.load_item_from(basename,
                                            "flatcentroid_sol_json")

    bottomup_solutions = flatcentroid_info["bottom_up_solutions"]

    if 1:

        orders = range(len(bottomup_solutions))

        from libs.apertures import Apertures
        ap =  Apertures(orders, bottomup_solutions)

        order_map2 = ap.make_order_map(mask_top_bottom=True)

        # from libs.storage_descriptions import FLAT_MASK_DESC
        # flat_mask = igr_storage.load1(FLAT_MASK_DESC,
        #                               flat_on_filenames[0])
        flat_mask = caldb.load_image(basename, "flat_mask")
        bias_mask = flat_mask & (order_map2 > 0)


        caldb.store_image(basename, "bias_mask", bias_mask)

        # from libs.products import PipelineImageBase, PipelineProducts
        # pp = PipelineProducts("")
        # from libs.storage_descriptions import BIAS_MASK_DESC
        # pp.add(BIAS_MASK_DESC,
        #        PipelineImageBase([], bias_mask))

        # flaton_basename = flat_on_filenames[0]
        # igr_storage.store(pp,
        #                   mastername=flaton_basename,
        #                   masterhdu=hdu)


def store_qa(helper, band, obsids_off, obsids_on):

    caldb = helper.get_caldb()
    basename = (band, obsids_on[0])

    # plot qa figures.

    if 1:
        from libs.process_flat import plot_trace_solutions
        from matplotlib.figure import Figure

        fig1 = Figure(figsize=[9, 4])

        flat_deriv = caldb.load_image(basename, "flat_deriv")
        trace_dict = caldb.load_item_from(basename, "flatcentroids_json")

        from libs.flat_qa import check_trace_order
        check_trace_order(flat_deriv, trace_dict, fig1)

        flat_normed = caldb.load_image(basename, "flat_normed")
        flatcentroid_sol_json = caldb.load_item_from(basename,
                                                     "flatcentroid_sol_json")

        from libs.flat_qa import plot_trace_solutions
        fig2, fig3 = plot_trace_solutions(flat_normed,
                                          flatcentroid_sol_json)

    # flatoff_basename = os.path.splitext(os.path.basename(flat_off_filenames[0]))[0]
    # flaton_basename = os.path.splitext(os.path.basename(flat_on_filenames[0]))[0]

    flatoff_basename = helper.get_basename(band, obsids_off[0])
    flaton_basename = helper.get_basename(band, obsids_on[0])

    igr_path = helper.igr_path

    if 1:
        from libs.qa_helper import figlist_to_pngs
        get_filename = igr_path.get_section_filename_base
        aperture_figs = get_filename("QA_PATH",
                                     "aperture_"+flaton_basename,
                                     "aperture_"+flaton_basename)

        figlist_to_pngs(aperture_figs, [fig1, fig2, fig3])


    # if 1: # now trace the orders

    #     #del trace_solution_products["bottom_up_solutions"]
    #     igr_storage.store(trace_solution_products,
    #                       mastername=flat_on_filenames[0],
    #                       masterhdu=flat_on_hdu_list[0])


def store_db(helper, band, obsids_off, obsids_on):

    caldb = helper.get_caldb()

    flatoff_basename = helper.get_basename(band, obsids_off[0])
    flaton_basename = helper.get_basename(band, obsids_on[0])

    # save db
    if 1:
        flatoff_db = caldb.load_db("flat_off")
        flatoff_db.update(band, flatoff_basename)

        flaton_db = caldb.load_db("flat_on")
        flaton_db.update(band, flaton_basename)

        # from libs.products import ProductDB
        # flatoff_db_name = get_filename("PRIMARY_CALIB_PATH",
        #                                "flat_off.db")

        # flatoff_db = ProductDB(flatoff_db_name)
        # #dbname = os.path.splitext(os.path.basename(flat_off_filenames[0]))[0]


        # flaton_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
        #                                                      "flat_on.db",
        #                                                      )
        # flaton_db = ProductDB(flaton_db_name)


def process_aux(helper, band, obsids_off, obsids_on):
    store_aux_data(helper, band, obsids_on)
    store_qa(helper, band, obsids_off, obsids_on)
    store_db(helper, band, obsids_off, obsids_on)


def process_band(utdate, recipe_name, band,
                 obsids_off, obsids_on, config_name):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    # STEP 1 :
    ## make combined image

    process_flat_off(helper, band, obsids_off)

    process_flat_on(helper, band, obsids_on)

    process_aux(helper, band, obsids_off, obsids_on)

    # make_combined_image(helper, band, obsids, mode=None)

if 0:
    # Step 2

    ## load simple-aperture (no order info; depends on

    extract_spectra(helper, band, obsids)
    ## aperture trace from Flat)

    ## extract 1-d spectra from ThAr


    # Step 3:
    ## compare to reference ThAr data to figure out orders of each strip
    ##  -  simple correlation w/ clipping

    identify_orders(helper, band, obsids)

    # Step 4:
    ##  - For each strip, measure x-displacement from the reference
    ##    spec. Fit the displacement as a function of orders.
    ##  - Using the estimated displacement, identify lines from the spectra.
    identify_lines(helper, band, obsids)


    # Step 6:

    ## load the reference echellogram, and find the transform using
    ## the identified lines.

    from find_affine_transform import find_affine_transform
    find_affine_transform(helper, band, obsids)

    from libs.transform_wvlsol import transform_wavelength_solutions
    transform_wavelength_solutions(helper, band, obsids)

    # Step 8:

    ## make order_map and auxilary files.

    save_figures(helper, band, obsids)

    save_db(helper, band, obsids)


if __name__ == "__main__":

    utdate = "20160226"
    obsids_off = range(11, 21)
    obsids_on = range(583, 593)

    recipe_name = "FLAT"

    band = "H"

    config_name = "../recipe.config"

    helper = RecipeHelper(config_name, utdate)

    process_band(utdate, recipe_name, band,
                 obsids_off, obsids_on, config_name)