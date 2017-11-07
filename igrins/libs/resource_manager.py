import re

from ..resource_manager import (ResourceStack, ResourceStackWithBasename,
                                ResourceDBSet)

from ..resource_manager.file_storage_igrins import get_storage
from ..resource_manager.resource_db_igrins \
    import get_igrins_db_factory

from .storage_descriptions import load_resource_def


class IgrinsBasenameHelper():
    p = re.compile(r"SDC(\w)_(\d+\w*)_(\d+)([^_]*)")
    p_obsid = re.compile(r"(\d+)(.*)")

    def __init__(self, obsdate, band):
        self.obsdate = obsdate
        self.band = band

    def to_basename(self, obsid):
        if isinstance(obsid, int):
            group_postfix = ""
        else:
            obsid_, group_postfix = self.p_obsid.match(obsid).groups()
            obsid = int(obsid_)

        return "SDC{band}_{obsdate}_{obsid:04d}{group_postfix}".format(obsdate=self.obsdate,
                                                                       band=self.band,
                                                                       obsid=obsid,
                                                                       group_postfix=group_postfix)

    def from_basename(self, basename):
        m = self.p.match(basename)
        return str(int(m.group(3))) + m.group(4)


def get_file_storage(config, resource_spec):
    return get_storage(config, resource_spec)


def get_resource_db(config, resource_spec):

    db_factory = get_igrins_db_factory(config, resource_spec)

    resource_def = load_resource_def()

    return ResourceDBSet(resource_spec,
                         db_factory, resource_def)


def get_resource_manager(config, resource_spec, basename_helper=None):

    base_storage = get_file_storage(config, resource_spec)

    from ..libs.item_convert import ItemConverter
    storage = ItemConverter(base_storage)

    resource_db = get_resource_db(config, resource_spec)

    if basename_helper is None:
        resource_manager = ResourceStack(resource_spec, storage,
                                         resource_db)
    else:
        resource_manager = ResourceStackWithBasename(resource_spec, storage,
                                                     resource_db,
                                                     basename_helper)

    return resource_manager


def get_igrins_resource_manager(config, resource_spec):
    obsdate, band = resource_spec

    basename_helper = IgrinsBasenameHelper(obsdate, band)

    rs = get_resource_manager(config, resource_spec,
                              basename_helper=basename_helper)

    return rs
