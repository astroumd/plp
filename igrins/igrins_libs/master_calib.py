import json
import os

from astropy.io import fits
import numpy as np
from six.moves import configparser as ConfigParser

def soft_link_loader(func):
    def wrapper(fn):
        try:
            return func(fn)
        except (json.decoder.JSONDecodeError, ):
            print("Loader error in function {} for file {}".format(func, fn))
            base_path = os.path.dirname(fn)
            with open(fn) as f:
                link_name = f.read()
                link_name = link_name.strip()
                fn = os.path.join(base_path, link_name)
            print("New filename: {}".format(fn))
            return func(fn)
    return wrapper


def get_master_calib_abspath(fn):
    return os.path.join("master_calib", fn)


@soft_link_loader
def json_loader(fn):
    # TODO: fix this throughout the code / master_cal.config so that the
    #  UTDATE variable is no longer necessary, and the soft links can be removed
    return json.load(open(fn))


@soft_link_loader
def fits_loader(fn):
    return fits.open(fn)

@soft_link_loader
def npy_loader(fn):
    return np.load(fn)


ref_loader_dict = {".json":json_loader,
                   ".fits":fits_loader,
                   ".npy":npy_loader}

def get_ref_loader(fn):
    fn1, ext = os.path.splitext(fn)

    loader = ref_loader_dict[ext]
    return loader


def query_ref_value_from_section(config, band, section, kind,
                                 ref_utdate=None, default=None):
    # if ref_utdate is None:
    #     ref_utdate = config.get("MASTER_CAL", "REFDATE")
    # master_cal_dir = config.get("MASTER_CAL", "MASTER_CAL_DIR")
    try:
        v = config.get(section, kind,
                       BAND=band)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        v = default
    return v

def query_ref_value(config, band, kind, ref_utdate=None):
    if ref_utdate is None:
        ref_utdate = config.get("MASTER_CAL", "REFDATE")
    master_cal_dir = config.get("MASTER_CAL", "MASTER_CAL_DIR")
    v = config.get("MASTER_CAL", kind,
                   MASTER_CAL_DIR=master_cal_dir,
                   REFDATE=ref_utdate,
                   BAND=band)
    return v


def query_ref_data_path(config, band, kind, ref_utdate=None):
    fn0 = query_ref_value(config, band, kind, ref_utdate=ref_utdate)

    fn = os.path.join(config.master_cal_dir, fn0)
    return fn


def load_ref_data(config, band, kind, ref_utdate=None):
    fn = query_ref_data_path(config, band, kind, ref_utdate=ref_utdate)
    loader = get_ref_loader(fn)
    return loader(fn)

