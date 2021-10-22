import os
import astropy.io.fits as pyfits


def get_first_science_hdu(hdu_list):
    if hdu_list[0].data is None:
        return hdu_list[1]
    else:
        return hdu_list[0]


def get_science_hdus(hdu_list):
    if hdu_list[0].data is None:
        return hdu_list[1:]
    else:
        return hdu_list[0:]


def open_fits(fn):

    if os.path.exists(fn):
        return pyfits.open(fn)

    fn_search_list = [fn]

    for gen_candidate, open_fits in candidate_generators:
        fn1 = gen_candidate(fn)
        if os.path.exists(fn1):
            return open_fits(fn1)

        fn_search_list.append(fn1)
    else:
        raise RuntimeError("No candidate files are found : %s" 
                           % fn_search_list)


