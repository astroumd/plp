from __future__ import print_function

import numpy as np
import scipy.ndimage as ni

from . import badpixel as bp


def get_flat_normalization(flat_on_off, bg_std, bpix_mask):

    lower_limit = bg_std*10

    flat_norm = bp.estimate_normalization_percentile(flat_on_off,
                                                     lower_limit, bpix_mask,
                                                     percentile=99.)
    # alternative normalization value
    # norm1 = bp.estimate_normalization(d, lower_limit, bpix_mask)

    return flat_norm

def estimate_bg_mean_std_deveny(flat, pad=4):
    '''Code to calculate background mean and stddev for Deveny'''

    #flat = flat[pad:-pad, pad:-pad]

    data = flat[5:44, 80:2000]

    flat_bg = np.mean(data)
    fwhm = np.std(data)*2*np.sqrt(2*np.log(2))

    return flat_bg, fwhm

def estimate_bg_mean_std(flat, pad=4, smoothing_length=150):

    flat = flat[pad:-pad, pad:-pad]

    flat_flat = flat[np.isfinite(flat)].flat
    flat_flat = flat_flat[flat_flat != 0] #added to remove stuff set to 0
    flat_sorted = np.sort(flat_flat)

    flat_gradient = ni.gaussian_filter1d(flat_sorted,
                                         smoothing_length, order=1)
    
    corr = False
    if np.min(flat_gradient) == 0:
        print("NJM: CORR, CHECK ON REAL DATA")
        fact = 0.1
        tmp = np.unique(flat_sorted)
        diff = tmp[1] - tmp[0]
        npts = len(flat_sorted)
        flat_sorted += np.random.randn(npts)*diff*fact
        flat_sorted = np.sort(flat_sorted)
        flat_gradient = ni.gaussian_filter1d(flat_sorted,
                                             smoothing_length, order=1)
        mean_tmp = 0
        std_tmp = diff*fact
        corr = True

    flat_sorted = flat_sorted[smoothing_length:]
    flat_dist = 1. / flat_gradient[smoothing_length:]

    over_half_mask = flat_dist > 0.5 * max(flat_dist)

    max_width_slice = max((sl.stop-sl.start, sl) for sl,
                          in ni.find_objects(over_half_mask))[1]

    flat_selected = flat_sorted[max_width_slice]

    l = len(flat_selected)
    indm = int(0.5 * l)
    # ind1, indm, ind2 = map(int, [0.05 * l, 0.5 * l, 0.95 * l])

    flat_bg = flat_selected[indm]

    fwhm = flat_selected[-1] - flat_selected[0]

    #plt.figure('FLAT SORTED RANDOM')
    #plt.plot(flat_sorted)
    #plt.figure('FLAT DIST')
    #plt.plot(flat_dist)
    #Correction for the noise added in
    if corr:
        fact = 2 * np.sqrt(2* np.log(2))
        std = fwhm / fact
        std_corr = np.sqrt(std**2 - std_tmp**2)
        fwhm_tmp = std_corr * fact
        flat_bg -= mean_tmp
        if np.isnan(fwhm):
            print("NAN FWHM")
            fwhm = 0.0001
    
    #print("SSS:", flat_bg, fwhm, flat_selected[0], flat_selected[-1])
    #plt.show()
    
    return flat_bg, fwhm


def get_flat_mask_auto(flat_bpix):
    # now we try to build a reasonable mask
    # start with a simple thresholded mask

    bg_mean, bg_fwhm = estimate_bg_mean_std(flat_bpix, pad=4,
                                            smoothing_length=150)
    with np.errstate(invalid="ignore"):
        flat_mask = (flat_bpix > bg_mean + bg_fwhm*3)

    # remove isolated dots by doing binary erosion
    m_opening = ni.binary_opening(flat_mask, iterations=2)
    # try to extend the mask with dilation
    m_dilation = ni.binary_dilation(m_opening, iterations=5)
        
    return m_dilation


def get_y_derivativemap(flat, flat_bpix, bg_std_norm,
                        max_sep_order=150, pad=50,
                        med_filter_size=(7, 7),
                        flat_mask=None, bound=None):

    """
    flat
    flat_bpix : bpix'ed flat
    """

    # 1d-derivatives along y-axis : 1st attempt
    # im_deriv = ni.gaussian_filter1d(flat, 1, order=1, axis=0)

    # 1d-derivatives along y-axis : 2nd attempt. Median filter first.

    # flat_deriv_bpix = ni.gaussian_filter1d(flat_bpix, 1,
    #                                        order=1, axis=0)

    # We also make a median-filtered one. This one will be used to make masks.
    flat_medianed = ni.median_filter(flat,
                                     size=med_filter_size)

    flat_deriv = ni.gaussian_filter1d(flat_medianed, 1,
                                      order=1, axis=0)
    if bound is not None:
        flat_deriv[:bound[0], :] = 0
        flat_deriv[bound[1]:, :] = 0

    # min/max filter

    print("MAX_SEP_ORDER:", max_sep_order)
    #max_sep_order = 60
    flat_max = ni.maximum_filter1d(flat_deriv, size=max_sep_order, axis=0)
    flat_min = ni.minimum_filter1d(flat_deriv, size=max_sep_order, axis=0)

    '''
    REMOVE
    import matplotlib.pyplot as plt
    plt.figure("FLAT")
    plt.imshow(flat)
    plt.figure("FLAT DERIV")
    plt.imshow(flat_deriv)
    plt.figure("FLAT MAX")
    plt.imshow(flat_max)
    plt.figure("FLAT MASK")
    plt.imshow(flat_mask)
    '''

    # mask for aperture boundray
    if pad is None:
        sl = slice()
    else:
        sl = slice(pad, -pad)

    flat_deriv_masked = np.zeros_like(flat_deriv)
    flat_deriv_masked[sl, sl] = flat_deriv[sl, sl]
  
    '''
    print("SSS:", np.shape(flat_deriv_masked), np.shape(flat_deriv), np.shape(flat_max))

    #x_idx = 2450
    x_idx = 2340
    plt.figure('MASK TEST')
    #plt.plot(flat_deriv_masked[:, x_idx], 'b')
    plt.plot(flat_deriv[:, x_idx], 'b')
    plt.plot(flat_max[:, x_idx], 'r')
    plt.plot(flat_max[:, x_idx]*0.1, 'r--')
    plt.plot(flat_max[:, x_idx]*0.5, 'r:')
    plt.plot(flat_min[:, x_idx], 'g')
    plt.plot(flat_min[:, x_idx]*0.1, 'g--')
    plt.plot(flat_min[:, x_idx]*0.5, 'g:')
    #plt.figure('MASK TEST2')
    #plt.plot(flat_deriv_masked[2450, :], 'b')
    #plt.plot(flat_max[2450, :], 'r')
    #plt.plot(flat_min[2450, :], 'g')
    '''

    flat_mask[:] = 1

    if flat_mask is not None:
        #flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5) & flat_mask
        #flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5) & flat_mask
        flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5) & flat_mask
        flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5) & flat_mask
        '''import matplotlib.pyplot as plt
        plt.figure("POS_MSK")
        plt.imshow(flat_deriv_pos_msk)
        plt.figure("NEG_MSK")
        plt.imshow(flat_deriv_neg_msk)
        plt.figure("FLAT_MIN * 0.5")
        plt.imshow(flat_min*0.5)
        plt.figure("FLAT_DERIV_MASKED")
        plt.imshow(flat_deriv_masked)
        plt.figure("FLAT_MASK")
        plt.imshow(flat_mask)
        plt.figure("TEST2")
        plt.imshow(flat_deriv_masked < flat_min * 0.5)
        plt.show()'''
    else:
        flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5)
        flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5)
    
    return dict(data=flat_deriv,  # _bpix,
                pos_mask=flat_deriv_pos_msk,
                neg_mask=flat_deriv_neg_msk,
                )


def mask_median_clip(y_ma, median_size=5, clip=1):
    """
    Subtract a median-ed singal from the original.
    Then, return a mask with out-of-sigma values clipped.
    """
    from scipy.stats.mstats import trima
    from scipy.signal import medfilt
    y_filtered = y_ma - medfilt(y_ma, median_size)
    y_trimmed = trima(y_filtered, (-clip, clip))
    return y_trimmed.mask


def find_nearest_object(mmp, im_labeled, slice_map, i, labels_center_column):
    """
    mmp : mask
    im_labeled : label
    i : object to be connected
    labels_center_column : known objects
    """
    thre = 40
    #thre = 100

    # threshold # of pixels (in y-direction) to detect adjacent object
    steps = [5, 10, 20, 40, 80]
    #steps = [5, 10, 20, 40, 80, 100, 120, 140, 160]

    sl_y, sl_x = slice_map[i]

    ## right side
    #ss = im_labeled[:, sl_x.stop-3:sl_x.stop].max(axis=1)
    #ss_msk = ni.maximum_filter1d(ss == i, thre)

    nx = len(mmp)

    if sl_x.stop < nx/2.:
        sl_x0 = sl_x.stop
        sl_x_pos = [sl_x.stop + s for s in steps]

        # right side of unlabeled object
        ss = im_labeled[:, sl_x.stop-3:sl_x.stop].max(axis=1)
        ss_msk = ni.maximum_filter1d(ss == i, thre)

        rhs=False
    else:
        sl_x0 = sl_x.start
        sl_x_pos = [sl_x.start - s for s in steps]

        # left side of unlabeled object
        ss = im_labeled[:, sl_x.start:sl_x.start+3].max(axis=1)
        ss_msk = ni.maximum_filter1d(ss == i, thre)

        rhs=True

    for pos in sl_x_pos:
        ss1 = im_labeled[:, pos]
        detected_ob = set(np.unique(ss1[ss_msk])) - set([0])
        #detected_ob = sorted(set(np.unique(ss1[ss_msk])) - set([0]))

        for ob_id in detected_ob:
            if ob_id in labels_center_column:
                sl = slice_map[ob_id][1]
                sl0 = slice_map[ob_id]
                #if rhs:
                #    if sl_y.stop > sl0[0].start:
                #        continue
                #else:
                #    if sl_y.start < sl0[0].stop:
                #        continue
                if sl0[1].start < sl_x0 < sl0[1].stop:
                    continue
                else:
                    return ob_id


def identify_horizontal_line(d_deriv, mmp, pad=20, bg_std=None,
                             thre_dx=30, cent_x=None, stitch_objects=False):
    """
    d_deriv : derivative (along-y) image
    mmp : mask
    order : polyfit order
    pad : padding around the boundary will be ignored
    bg_std : if given, derivative smaller than bg_std will be suppressed.
             This will affect faint signal near the chip boundary
    stitch_objects : bool, whether to try and stitch labels outside the center to labels in
                     the center

    Masks will be derived from mmp, and peak values of d_deriv will be
    fitted with polynomical of given order.

    We first limit the area between
       1024 - thre_dx > x > 1024 + thre_dx

    and identify objects from the mask whose x-slice is larger than thre_dx.
    This will identify at most one object per order. For objects not included
    in this list, we find nearest object and associate it to them.
    """
    ny, nx = d_deriv.shape

    # We first identify objects
    im_labeled, label_max = ni.label(mmp)
    label_indx = np.arange(1, label_max+1, dtype="i")
    objects_found = ni.find_objects(im_labeled)
    
    slice_map = dict(zip(label_indx, objects_found))

    # We only traces solutions that are detected in the centeral colmn

    # label numbers along the central column

    # from itertools import groupby
    # labels_center_column = [i for i, _ in groupby(im_labeled[:,nx/2]) if i>0]

    # thre_dx = 30
    if cent_x is None:
        cent_x = nx//2
    #center_cut = im_labeled[:, nx//2-thre_dx:nx//2+thre_dx]
    center_cut = im_labeled[:, cent_x-thre_dx:cent_x+thre_dx]
    labels_ = list(set(np.unique(center_cut)) - set([0]))

    if True:  # remove false detections
        sl_subset = [slice_map[l][1] for l in labels_]
        mm = [(sl1.stop - sl1.start) > thre_dx for sl1 in sl_subset]
        labels1 = [l1 for l1, m1 in zip(labels_, mm) if m1]

    # for i in labels_:
    #     if i not in labels1:
    #         center_cut[center_cut == i] = 0

    labels_center_column = sorted(labels1)

    # remove objects with small area
    s = ni.measurements.sum(mmp, labels=im_labeled,
                            index=labels_center_column)

    labels_center_column = np.array(labels_center_column)[s > 0.1 * s.max()]

    # try to stitch undetected object to center ones.
    undetected_labels_ = [i for i in range(1, label_max+1)
                          if i not in labels_center_column]
    s2 = ni.measurements.sum(mmp, labels=im_labeled,
                             index=undetected_labels_)

    undetected_labels = np.array(undetected_labels_)[s2 > 0.1 * s.max()]

    slice_map_update_required = False

    if stitch_objects:
        rerun = True
        while rerun:
            attached_labels = []
            rerun = False
            for i in undetected_labels:
                ob_id = find_nearest_object(mmp, im_labeled,
                                            slice_map, i, labels_center_column)

                if ob_id:
                    im_labeled[im_labeled == i] = ob_id
                    slice_map_update_required = True

                    objects_found = ni.find_objects(im_labeled)
                    slice_map = dict(zip(label_indx, objects_found))

                    rerun = True
                    attached_labels.append(i)

            undetected_labels = np.setdiff1d(undetected_labels, attached_labels)


    if slice_map_update_required:
        objects_found = ni.find_objects(im_labeled)
        slice_map = dict(zip(label_indx, objects_found))

    # im_labeled is now updated

    y_indices = np.arange(ny)
    x_indices = np.arange(nx)

    centroid_list = []
    print("SETTING BG_STD TO NONE:", bg_std)
    bg_std = None
    for indx in labels_center_column:

        sl = slice_map[indx]

        y_indices1 = y_indices[sl[0]]
        x_indices1 = x_indices[sl[1]]

        # mask for line to trace
        feature_msk = im_labeled[sl] == indx

        # nan for outer region.
        feature = d_deriv[sl].copy()
        feature[~feature_msk] = np.nan

        # measure centroid
        yc = np.nansum(y_indices1[:, np.newaxis] * feature, axis=0)
        ys = np.nansum(feature, axis=0)
        yn = np.sum(np.isfinite(feature), axis=0)

        with np.errstate(invalid="ignore"):
            yy = yc/ys

            msk = mask_median_clip(yy) | ~np.isfinite(yy)
           
            # we also clip whose derivative is smaller than bg_std
            # This suprress the lowest order of K band
            if bg_std is not None:
                msk = msk | (ys/yn < bg_std)
                # msk = msk | (ys/yn < 0.0006 + 0.0003)

            # mask out columns with # of valid pixel is too many
            # number 10 need to be fixed - JJL
            msk = msk | (yn > 10)
            #zzz

        centroid_list.append((x_indices1,
                              np.ma.array(yy, mask=msk)))

    return centroid_list


def trace_aperture_chebyshev(xy_list, domain):
    """
    a list of (x_array, y_array).

    y_array must be a masked array
    """
    import numpy.polynomial.chebyshev as cheb

    domain_order = {}
    # we first fit the all traces with 2d chebyshev polynomials
    x_list, o_list, y_list = [], [], []
    for o, (x, y) in enumerate(xy_list):
        if hasattr(y, "mask"):
            msk = ~y.mask & np.isfinite(y.data)
            y = y.data
        else:
            msk = np.isfinite(np.array(y, "d"))
        x1 = np.array(x)[msk]
        x_list.append(x1)
        o_list.append(np.zeros(len(x1))+o)
        y_list.append(np.array(y)[msk])
        domain_order[o] = [np.min(x1), np.max(x1)]
    n_o = len(xy_list)

    if n_o == 1:
        n_o += 1

    from astropy.modeling import fitting  # models, fitting
    from astropy.modeling.polynomial import Chebyshev2D
    x_degree, y_degree = 4, 5
    #NJM: Added because Deveny only has a single order so
    #we can't fit with y_degree=5
    #if len(np.unique(o_list)) < y_degree:
    if len(o_list) < y_degree:
        y_degree = len(np.unique(o_list)) - 1
    p_init = Chebyshev2D(x_degree, y_degree,
                         x_domain=domain, y_domain=[0, n_o-1])
    fit_p = fitting.LinearLSQFitter()

    xxx, ooo, yyy = (np.concatenate(x_list),
                     np.concatenate(o_list),
                     np.concatenate(y_list))
    p = fit_p(p_init, xxx, ooo, yyy)

    for ii in range(3):  # number of iteration
        mmm = np.abs(yyy - p(xxx, ooo)) < 1
        # This need to be fixed with actual estimation of sigma.

        p = fit_p(p_init, xxx[mmm], ooo[mmm], yyy[mmm])

    # Now we need to derive a 1d chebyshev for each order.  While
    # there should be an analytical way, here we refit the trace for
    # each order using the result of 2d fit.

    xx = np.arange(domain[0], domain[1])
    oo = np.zeros_like(xx)

    ooo = [o[0] for o in o_list]

    def _get_f(o0):
        #y_m = p(xx, oo+o0)
        #f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        xx = np.arange(domain_order[o][0], domain_order[0][1])
        oo = np.zeros_like(xx)
        y_m = p(xx, oo+o0)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain_order[o])
        return f

    f_list = []
    f_list = [_get_f(o0) for o0 in ooo]
    domain_list = [domain_order[o0] for o0 in ooo]

    # def _get_f_old(next_orders, y_thresh):
    #     oi = next_orders.pop(0)
    #     y_m = p(xx, oo+oi)
    #     f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
    #     if next_orders:  # if not the last order
    #         if np.all(y_thresh(y_m)):
    #             print("all negative at ", oi)
    #             next_orders = next_orders[:1]

    #     return oi, f, next_orders

    def _get_f(next_orders, y_thresh):
        oi = next_orders.pop(0)
        oo = np.zeros_like(xx)
        y_m = p(xx, oo+oi)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        if np.all(y_thresh(y_m)):
            # print("all negative at ", oi)
            next_orders = []

        return oi, f, next_orders

    # go down in order
    f_list_down = []
    o_list_down = []
    domain_list_down = []

    #If we only have one order (Deveny), we cannot extrapolate to orders above/below the fit as
    #we have no info on how the order value affects the y position
    if len(f_list) == 1:
        go_down_orders = False
        go_up_orders = False
    else:
        go_down_orders = [ooo[0] - _oi for _oi in range(1, 5)]
        go_up_orders = [ooo[-1]+_oi for _oi in range(1, 5)]
    
    while go_down_orders:
        xx = np.arange(domain_order[ooo[0]][0], domain_order[ooo[0]][1])
        oi, f, go_down_orders = _get_f(go_down_orders,
                                       y_thresh=lambda y_m: y_m < domain[0])
        f_list_down.append(f)
        o_list_down.append(oi)
        domain_list_down.append(domain_order[ooo[0]])

    f_list_up = []
    o_list_up = []
    domain_list_up = []

    while go_up_orders:
        xx = np.arange(domain_order[ooo[-1]][0], domain_order[ooo[-1]][1])
        oi, f, go_up_orders = _get_f(go_up_orders,
                                     y_thresh=lambda y_m: y_m > domain[-1])
        f_list_up.append(f)
        o_list_up.append(oi)
        domain_list_up.append(domain_order[ooo[-1]])

    return f_list, f_list_down[::-1] + f_list + f_list_up, domain_list_down[::-1] + domain_list + domain_list_up


def get_matched_slices(yc_down_list, yc_up_list):

    mid_indx_down = len(yc_down_list) // 2

    mid_indx_up = np.searchsorted(yc_up_list, yc_down_list[mid_indx_down])

    n_lower = min(mid_indx_down, mid_indx_up)

    n_upper = min(len(yc_down_list) - mid_indx_down,
                  len(yc_up_list) - mid_indx_up)

    slice_down = slice(mid_indx_down - n_lower, mid_indx_down + n_upper)
    slice_up = slice(mid_indx_up - n_lower, mid_indx_up + n_upper)

    return slice_down, slice_up


def trace_centroids_chebyshev(centroid_bottom_list,
                              centroid_up_list,
                              domain, nx, ref_x=None):
    #TODO: See difference in domain and nx variables

    if ref_x is None:
        ref_x = 0.5 * (domain[0] + domain[-1])

    _ = trace_aperture_chebyshev(centroid_bottom_list,
                                 domain)
    sol_bottom_list, sol_bottom_list_full, domain_bottom_list = _

    _ = trace_aperture_chebyshev(centroid_up_list,
                                 domain)
    sol_up_list, sol_up_list_full, domain_up_list = _
    
    yc_down_list = [s(ref_x) for s in sol_bottom_list_full]
    # lower-boundary list
    yc_up_list = [s(ref_x) for s in sol_up_list_full]
    # upper-boundary list

    # yc_down_list[1] should be the 1st down-boundary that is not
    # outside the detector

    #If we only have a single order (Deveny), we skip all this filtering stuff
    if len(yc_down_list) == 1:
        sol_bottom_up_list_full_filtered = [(sol_bottom_list[0], sol_up_list[0])]
        #domain_bottom_up_list_filtered = [(domain_bottom_list[0], domain_up_list[0])]
        d0 = max(domain_bottom_list[0][0], domain_up_list[0][0])
        d1 = min(domain_bottom_list[0][1], domain_up_list[0][1])
        domain_bottom_up_list_filtered = [(d0, d1)]
        centroid_bottom_up_list = centroid_bottom_list, centroid_up_list
        sol_bottom_up_list = sol_bottom_list, sol_up_list
        return (sol_bottom_up_list_full_filtered,
                sol_bottom_up_list, centroid_bottom_up_list,
                domain_bottom_up_list_filtered)

    indx_down_bottom = np.searchsorted(yc_down_list, yc_up_list[1])
    indx_up_top = np.searchsorted(yc_up_list, yc_down_list[-2],
                                  side="right")

    sol_bottom_up_list_full = zip(sol_bottom_list_full[indx_down_bottom-1:-1],
                                  sol_up_list_full[1:indx_up_top+1])

    domain_bottom_up_list_full = zip(domain_bottom_list[indx_down_bottom-1:-1],
                                     domain_up_list[1:indx_up_top+1])

    slice_down, slice_up = get_matched_slices(yc_down_list, yc_up_list)

    sol_bottom_up_list_full = zip(sol_bottom_list_full[slice_down],
                                  sol_up_list_full[slice_up])

    domain_bottom_up_list_full = zip(domain_bottom_list[slice_down],
                                     domain_up_list[slice_up])

    domain_list_full = []
    for domain_bottom, domain_up in domain_bottom_up_list_full:
        d0 = max(domain_bottom[0], domain_up[0])
        d1 = min(domain_bottom[1], domain_up[1])
        domain_list_full.append((d0, d1))

    # check if the given order has pixels in the detector
    x = np.arange(nx)
    sol_bottom_up_list_full_filtered = []
    domain_bottom_up_list_filtered = []
    for s_tmp, d_tmp in zip(sol_bottom_up_list_full, domain_list_full):
        s_bottom, s_up = s_tmp
        if max(s_up(x)) > 0. and min(s_bottom(x)) < nx:
            sol_bottom_up_list_full_filtered.append((s_bottom, s_up))
            domain_bottom_up_list_filtered.append(d_tmp)
    # print sol_bottom_up_list_full

    sol_bottom_up_list = sol_bottom_list, sol_up_list
    centroid_bottom_up_list = centroid_bottom_list, centroid_up_list
    # centroid_bottom_up_list = []

    return (sol_bottom_up_list_full_filtered,
            sol_bottom_up_list, centroid_bottom_up_list,
            domain_bottom_up_list_filtered)


def get_smoothed_order_spec(s):
    s = np.array(s)
    k1, k2 = np.nonzero(np.isfinite(s))[0][[0, -1]]
    s1 = s[k1:k2+1]

    s0 = np.empty_like(s)
    s0.fill(np.nan)
    s0[k1:k2+1] = ni.median_filter(s1, 40)
    return s0


def get_order_boundary_indices(s1, s0=None, nx=4096):
    # x = np.arange(len(s))

    # select finite number only. This may happen when orders go out of
    # chip boundary.
    s1 = np.array(s1)
    # k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]

    with np.errstate(invalid="ignore"):
        nonzero_indices = np.nonzero(s1 > 0.05)[0]  # [[0, -1]]

    # return meaningless indices if non-zero spectra is too short
    with np.errstate(invalid="ignore"):
        if len(nonzero_indices) < 5:
            return 4, 4

    k1, k2 = nonzero_indices[[0, -1]]
    k1 = max(k1, 4)
    k2 = min(k2, nx-5)
    s = s1[k1:k2+1]

    if s0 is None:
        s0 = get_smoothed_order_spec(s)
    else:
        s0 = s0[k1:k2+1]

    mm = s > max(s) * 0.05
    dd1, dd2 = np.nonzero(mm)[0][[0, -1]]

    # mask out absorption feature
    smooth_size = 20
    # s_s0 = s-s0
    # s_s0_std = s_s0[np.abs(s_s0) < 2.*s_s0.std()].std()

    # mmm = s_s0 > -3.*s_s0_std

    s1 = ni.gaussian_filter1d(s0[dd1:dd2], smooth_size, order=1)
    # x1 = x[dd1:dd2]

    # s1r = s1 # ni.median_filter(s1, 100)

    s1_std = s1.std()
    s1_std = s1[np.abs(s1) < 2.*s1_std].std()

    s1[np.abs(s1) < 2.*s1_std] = np.nan

    indx_center = int(len(s1)*.5)

    left_half = s1[:indx_center]
    if np.any(np.isfinite(left_half)):
        i1 = np.nanargmax(left_half)
        a_ = np.where(~np.isfinite(left_half[i1:]))[0]
        if len(a_):
            i1r = a_[0]
        else:
            i1r = 0
        i1 = dd1+i1+i1r  # +smooth_size
    else:
        i1 = dd1

    right_half = s1[indx_center:]
    if np.any(np.isfinite(right_half)):
        i2 = np.nanargmin(right_half)
        a_ = np.where(~np.isfinite(right_half[:i2]))[0]

        if len(a_):
            i2r = a_[-1]
        else:
            i2r = i2
        i2 = dd1+indx_center+i2r
    else:
        i2 = dd2

    return k1+i1, k1+i2


def get_finite_boundary_indices(s1):
    # select finite number only. This may happen when orders go out of
    # chip boundary.
    s1 = np.array(s1)
    with np.errstate(invalid="ignore"):
        nonzero_indices = np.nonzero(s1 > 0.)[0]  # [[0, -1]]

    print("HARDCODING 4095 GET_FINITE_BOUNDARY_INDICES")
    k1, k2 = nonzero_indices[[0, -1]]
    k1 = max(k1, 4)
    k2 = min(k2, 4095-4)
    return k1, k2

#NOTE: Functions readded in for qa plots for register
def prepare_order_trace_plot(s_list, row_col=(3, 2)):

    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1 import Grid
    #TODO: redo this
    #from axes_grid_patched import Grid

    row, col = row_col

    n_ax = len(s_list)
    n_f, n_remain = divmod(n_ax, row*col)
    if n_remain:
        n_ax_list = [row*col]*n_f + [n_remain]
    else:
        n_ax_list = [row*col]*n_f


    i_ax = 0

    fig_list = []
    ax_list = []
    for n_ax in n_ax_list:
        fig = Figure()
        fig_list.append(fig)

        if n_ax < row*col:
            n_ax = row*col

        grid = Grid(fig, 111, (row, col), ngrids=n_ax, share_x=True)

        sl = slice(i_ax, i_ax+n_ax)
        for s, ax in zip(s_list[sl], grid):
            ax_list.append(ax)

        i_ax += n_ax

    return fig_list, ax_list

def check_order_trace1(ax, x, s, i1i2):
    x = np.arange(len(s))
    ax.plot(x, s)
    i1, i2 = i1i2
    ax.plot(np.array(x)[[i1, i2]], np.array(s)[[i1,i2]], "o")

def check_order_trace2(ax, x, p):
    ax.plot(x, p(x))

def get_order_flat1d(s, i1=None, i2=None):

    s = np.array(s)
    k1, k2 = np.nonzero(np.isfinite(s))[0][[0, -1]]
    s1 = s[k1:k2+1]


    if i1 is None:
        i1 = 0
    else:
        i1 -= k1

    if i2 is None:
        i2 = len(s1)
    else:
        i2 -= k1

    x = np.arange(len(s1))

    if 1:
        t_list = []
        if i1 > 10:
            t_list.append([x[1],x[i1]])
        else:
            t_list.append([x[1]])

        t_list.append(np.linspace(x[i1]+10, x[i2-1]-10, 10))
        if i2 < len(s) - 10:
            t_list.append([x[i2], x[-2]])
        else:
            t_list.append([x[-2]])

        t= np.concatenate(t_list)

        # s0 = ni.median_filter(s, 40)
        from scipy.interpolate import LSQUnivariateSpline
        p = LSQUnivariateSpline(x,
                                s1,
                                t, bbox=[0, len(s1)-1])

        def p0(x, k1=k1, k2=k2, p=p):
            msk = (k1 <= x) & (x <= k2)
            r = np.empty(len(x), dtype="d")
            r.fill(np.nan)
            r[msk] = p(x[msk])
            return r

    return p0
