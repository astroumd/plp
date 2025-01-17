import numpy as np
import itertools
from scipy.interpolate import interp1d
import scipy.spatial as spatial
import operator
import json
import os


def match_lines1_pixel(cent_list, ref_pix_list):
    """
    """

    if len(cent_list):

        # find nearest matches

        kdtree = spatial.KDTree(cent_list.reshape([-1,1]))
        dists, indices = kdtree.query(ref_pix_list.reshape([-1,1]))
        return cent_list[indices], dists

    else:
        cent_list = np.empty((len(ref_pix_list),), dtype="d")
        dists = np.empty((len(ref_pix_list),), dtype="d")
        cent_list.fill(np.nan)
        dists.fill(np.nan)

        return cent_list, dists


def match_lines1_pix(s, ref_pix_list):
    """

    returns matched_indices, matched_fit_params, matched_distances

    matched_indices : indices of lines (from ref_line_list) that are
                      associated with line features in the given spectra

    matched_distances : distances in pixel

    """

    # find centroids of s
    from .find_peak import find_peaks
    sol_list = find_peaks(s, sigma=3)
    cent_list = np.array([sol[0] for sol in sol_list if np.isfinite(sol[0])])

    cent_list, dists = match_lines1_pixel(cent_list, ref_pix_list)

    return cent_list, dists



def match_lines1(s, wvl, ref_line_list):
    """

    returns matched_indices, matched_fit_params, matched_distances

    matched_indices : indices of lines (from ref_line_list) that are
                      associated with line features in the given spectra

    matched_distances : distances in pixel

    """

    # find centroids of s
    from .find_peak import find_peaks
    sol_list = find_peaks(s, sigma=3)
    cent_list = np.array([sol[0] for sol in sol_list])

    wvl2pix = interp1d(wvl, np.arange(len(wvl)))
    ref_pix_list = wvl2pix(ref_line_list)

    cent_list, dists = match_lines1_pixel(cent_list, ref_pix_list)

    return cent_list, dists




def match_lines2(s, wvl, ref_line_list):
    """

    returns matched_indices, matched_fit_params, matched_distances

    matched_indices : indices of lines (from ref_line_list) that are
                      associated with line features in the given spectra

    matched_distances : distances in pixel

    """

    # find centroids of s
    from .find_peak import find_peaks
    sol_list = find_peaks(s, sigma=3)
    cent_list = np.array([sol[0] for sol in sol_list])

    # define transform from lambda to pixel
    wvl2pix = interp1d(wvl, np.arange(len(wvl)))

    ref_pix_list = wvl2pix(ref_line_list)


    # find nearest matches

    kdtree = spatial.KDTree(ref_pix_list.reshape([-1,1]))
    dists, indices = kdtree.query(cent_list.reshape([-1,1]))

    # filter out multiple hits. Only the nearest one remains.
    filtered_indices = []
    for k, l in itertools.groupby(zip(indices,
                                      sol_list, dists),
                                  operator.itemgetter(0)):
        l = list(l)
        i = np.argmin([l1[-1] for l1 in l])
        filtered_indices.append(l[i])

    matched_indices =  [s_[0] for s_ in filtered_indices]
    matched_fit_params = [s_[1] for s_ in filtered_indices]
    matched_distances = [s_[2] for s_ in filtered_indices]

    return matched_indices, matched_fit_params, matched_distances

def match_lines(s, wvl, line_list_all):
    imin = np.searchsorted(line_list_all, wvl[0])
    imax = np.searchsorted(line_list_all, wvl[-1])

    # matched_indices, matched_fit_params, matched_distances = \
    #                  match_lines2(s, wvl, line_list_all[imin:imax])
    # matched_indices = np.array(matched_indices) + imin

    centroids, matched_distances = match_lines1(s, wvl,
                                                line_list_all[imin:imax])
    matched_indices = imin + np.arange(len(matched_distances))

    return matched_indices, centroids, matched_distances

def filter_distances(indices, distances, thresh):
    return indices[distances < thresh]


