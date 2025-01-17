import numpy as np

from ..pipeline.steps import Step

from ..procedures.readout_pattern_guard import remove_pattern_from_guard
from ..procedures.procedure_dark import (apply_rp_2nd_phase,
                                         apply_rp_3rd_phase)

from ..procedures.ro_pattern_fft import (get_amp_wise_rfft,
                                         make_model_from_rfft)
from .gui_combine import setup_gui, factory_pattern_remove_n_smoothed


def _get_combined_image(obsset):
    # Should not use median, Use sum.
    data_list = [hdu.data for hdu in obsset.get_hdus()]

    return np.sum(data_list, axis=0)


def remove_pattern(data_minus, mask=None, remove_level=1,
                   remove_amp_wise_var=True, nx=4096):

    #NOTE: No pattern removal for RIMAS or DEVENY
    if nx != 2048:
        return data_minus

    d1 = remove_pattern_from_guard(data_minus, nx=nx)

    if remove_level == 2:
        d2 = apply_rp_2nd_phase(d1, mask=mask)
    elif remove_level == 3:
        d2 = apply_rp_2nd_phase(d1, mask=mask)
        d2 = apply_rp_3rd_phase(d2)
    else:
        d2 = d1

    if remove_amp_wise_var:
        c = get_amp_wise_rfft(d2)

        ii = select_k_to_remove(c)
        print(ii)
        # ii = [9, 6]

        new_shape = (32, 64, nx)
        mm = np.zeros(new_shape)

        for i1 in ii:
            mm1 = make_model_from_rfft(c, slice(i1, i1+1))
            mm += mm1[:, np.newaxis, :]

        ddm = mm.reshape((-1, nx))

        return d2 - ddm

    else:
        return d2


def select_k_to_remove(c, n=2):
    ca = np.abs(c)
    # k = np.median(ca, axis=0)[1:]  # do no include the 1st column
    k = np.percentile(ca, 95, axis=0)[1:]  # do no include the 1st column
    # print(k[:10])
    x = np.arange(1, 1 + len(k))
    msk = (x < 5) | (15 < x)  # only select k from 5:15

    # polyfit with 5:15 data
    p = np.polyfit(np.log10(x[msk]), np.log10(k[msk]), 2,
                   w=1./x[msk])
    # p = np.polyfit(np.log10(x[msk][:30]), np.log10(k[msk][:30]), 2,
    #                w=1./x[msk][:30])
    # print(p)

    # sigma from last 256 values
    ss = np.std(np.log10(k[-256:]))

    # model from p with 3 * ss
    y = 10.**(np.polyval(p, np.log10(x)))

    di = 5
    dly = np.log10(k/y)[di:15]

    # select first two values above 3 * ss
    ii = np.argsort(dly)
    yi = [di + i1 + 1for i1 in ii[::-1][:n] if dly[i1] > 3 * ss]

    return yi


def get_combined_images(obsset,
                        allow_no_b_frame=False):

    ab_mode = obsset.recipe_name.endswith("AB")

    obsset_a = obsset.get_subset("A", "ON")
    obsset_b = obsset.get_subset("B", "OFF")

    na, nb = len(obsset_a.obsids), len(obsset_b.obsids)

    if ab_mode and (na != nb):
        if na > nb:
            obsids = obsset_a.obsids[:nb]
            obsset_a = obsset_a.get_subset_obsid(obsids)
        else:
            obsids = obsset_b.obsids[:na]
            obsset_b = obsset_b.get_subset_obsid(obsids)
        print("For AB nodding, number of A and B should match!")
        print("However we are removing A or B obsids until na == nb")

        #raise RuntimeError("For AB nodding, number of A and B should match!")

    if na == 0:
        raise RuntimeError("No A Frame images are found")

    if nb == 0 and not allow_no_b_frame:
        raise RuntimeError("No B Frame images are found")

    if nb == 0:
        a_data = _get_combined_image(obsset_a)
        data_minus = a_data

    else:  # nb > 0
        # a_b != 1 for the cases when len(a) != len(b)
        a_b = float(na) / float(nb)

        a_data = _get_combined_image(obsset_a)
        b_data = _get_combined_image(obsset_b)

        data_minus = a_data - a_b * b_data

    if nb == 0:
        data_plus = a_data
    else:
        data_plus = (a_data + (a_b**2)*b_data)

    return data_minus, data_plus


def get_variances(data_minus, data_plus, gain, nx=4096, ny=4096):

    """
    Return two variances.
    1st is variance without poisson noise of source added. This was
    intended to be used by adding the noise from simulated spectra.
    2nd is the all variance.

    """
    from igrins.procedures.procedure_dark import get_per_amp_stat

    guards = data_minus[:, [0, 1, 2, 3, -4, -3, -2, -1]]

    namp = ny // 64

    qq = get_per_amp_stat(guards, namp=namp)
 
    s = np.array(qq["stddev_lt_threshold"]) ** 2

    variance_per_amp = np.repeat(s, 64*nx).reshape((-1, nx))
    
    variance = variance_per_amp + np.abs(data_plus)/gain

    return variance_per_amp, variance


def run_interactive(obsset,
                    data_minus_raw, data_plus, bias_mask,
                    remove_level, remove_amp_wise_var):
    import matplotlib.pyplot as plt
    # from astropy_smooth import get_smoothed
    # from functools import lru_cache

    get_im = factory_pattern_remove_n_smoothed(remove_pattern,
                                               data_minus_raw,
                                               bias_mask)

    fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

    vmin, vmax = -30, 30
    # setup figure guis

    obsdate, band = obsset.get_resource_spec()
    obsid = obsset.master_obsid

    status = dict(to_save=False)

    def save(*kl, status=status):
        status["to_save"] = True
        plt.close(fig)
        # print("save")
        # pass

    ax.set_title("{}-{:04d} [{}]".format(obsdate, obsid, band))

    # add callbacks
    d2 = get_im(1, False, False)
    im = ax.imshow(d2, origin="lower", interpolation="none")
    im.set_clim(vmin, vmax)

    box, get_params = setup_gui(im, vmin, vmax,
                                get_im, save)

    plt.show()
    params = get_params()
    params.update(status)

    return params


def make_combined_images(obsset, allow_no_b_frame=False,
                         remove_level=2,
                         remove_amp_wise_var=False,
                         interactive=False,
                         cache_only=False):

    if remove_level == "auto":
        remove_level = 2

    if remove_amp_wise_var == "auto":
        remove_amp_wise_var = False

    _ = get_combined_images(obsset,
                            allow_no_b_frame=allow_no_b_frame)
    data_minus_raw, data_plus = _
    bias_mask = obsset.load_resource_for("bias_mask")

    if interactive:
        params = run_interactive(obsset,
                                 data_minus_raw, data_plus, bias_mask,
                                 remove_level, remove_amp_wise_var)

        print("returned", params)
        if not params["to_save"]:
            print("canceled")
            return

        remove_level = params["remove_level"]
        remove_amp_wise_var = params["amp_wise"]
    
    nx = obsset.detector.nx
    ny = obsset.detector.ny

    d2 = remove_pattern(data_minus_raw, mask=bias_mask,
                        remove_level=remove_level,
                        remove_amp_wise_var=remove_amp_wise_var,
                        nx=nx)

    dp = remove_pattern(data_plus, remove_level=1,
                        remove_amp_wise_var=False,
                        nx=nx)
    
    gain = float(obsset.rs.query_ref_value("GAIN"))
    
    if hasattr(obsset.detector, 'npad_p'):
        #print("FIX DEVENY VARIANCE CALCULATION")
        #print("Variance only works with padding in positive y direction")
        y1 = obsset.detector.ny0 + obsset.detector.npad_m
        #ny_var = 2**(int(np.log2(y1)))
        ny_var = y1 // 64 * 64
        y0 = y1 - ny_var
        d2_tmp = d2[y0:y1, :]
        dp_tmp = dp[y0:y1, :]

        variance_map0, variance_map = get_variances(d2_tmp, dp_tmp, gain, nx=nx, ny=ny_var)
    else:
        variance_map0, variance_map = get_variances(d2, dp, gain, nx=nx, ny=ny)
    
    if obsset.detector.name == 'deveny':
        variance_map0b = np.zeros_like(data_minus_raw)
        variance_mapb = np.zeros_like(data_minus_raw)
        
        variance_map0b[y0:y1, :] = variance_map0
        variance_mapb[y0:y1, :] = variance_map
        
        variance_map0b[:y0, :] = variance_map0[0, :][None, :]
        variance_map0b[y1:, :] = variance_map0[-1, :][None, :]
        
        variance_mapb[:y0, :] = variance_map[0, :][None, :]
        variance_mapb[y1:, :] = variance_map[-1, :][None, :]

        variance_map0 = variance_map0b
        variance_map = variance_mapb

    hdul = obsset.get_hdul_to_write(([], d2))

    obsset.store("combined_image1", data=hdul, cache_only=cache_only)

    hdul = obsset.get_hdul_to_write(([], variance_map0))
    obsset.store("combined_variance0", data=hdul, cache_only=cache_only)

    hdul = obsset.get_hdul_to_write(([], variance_map))
    obsset.store("combined_variance1", data=hdul, cache_only=cache_only)


steps = [Step("Make Combined Image", make_combined_images,
              interactive=False,
              remove_level="auto", remove_amp_wise_var="auto")]

