class ObsSet(object):
    def __init__(self, caldb, recipe_name, band, obsids, frametypes):
        self.caldb = caldb
        self.recipe_name = recipe_name
        self.band = band
        self.obsids = obsids
        self.frametypes = frametypes
        self.basename = self.caldb._get_basename((self.band, self.obsids[0]))

    def get_base_info(self):
        return self.caldb.get_base_info(self.band, self.obsids)

    def get_frames(self):
        pass

    def get_obsids(self, frametype=None):
        if frametype is None:
            return self.obsids
        else:
            obsids = [obsid for o, f \
                      in zip(self.obsids, self.frametypes) if f == frametype]

            return obsids

    def get_subset(self, frametype):
        obsids = [o for o, f in zip(self.obsids, self.frametypes)
                  if f == frametype]
        frametypes = [frametype] * len(obsids)

        return ObsSet(self.caldb, self.recipe_name, self.band, 
                      obsids, frametypes)

    def get_data_list(self):

        _ = self.get_base_info()
        filenames = _[0]

        from igrins.libs.load_fits import load_fits_data
        hdu_list = [load_fits_data(fn_) for fn_ in filenames]

        return [hdu.data for hdu in hdu_list]

    def load_db(self, db_name):
        return self.caldb.load_db(db_name)


    def query_item_path(self, item_type_or_desc,
                        basename_postfix=None, subdir=None):
        return self.caldb.query_item_path(self.basename, item_type_or_desc,
                                          basename_postfix=basename_postfix,
                                          subdir=subdir)

    def load_item(self, itemtype,
                  basename_postfix=None):
        return self.caldb.load_item_from(self.basename, itemtype,
                                         basename_postfix=basename_postfix)

    def load_image(self, item_type):
        "similar to load_item, but returns the image as a numpy.array"
        return self.caldb.load_image(self.basename, item_type)

    def _load_item_from(self, item_type_or_desc,
                        basename_postfix=None):
        return self.caldb.load_item_from(self, self.basename,
                                         item_type_or_desc,
                                         basename_postfix=basename_postfix)

    def store_dict(self, item_type, data):
        return self.caldb.store_dict(self.basename, item_type, data)

    def store_image(self, item_type, data, 
                    header=None, card_list=None):
        return self.caldb.store_image(self.basename,
                                      item_type=item_type, data=data,
                                      header=header, card_list=card_list)

    def store_multi_images(self, item_type, hdu_list,
                           basename_postfix=None):
        self.caldb.store_multi_image(self.basename,
                                     item_type, hdu_list,
                                     basename_postfix=basename_postfix)


    def load_ref_data(self, kind):
        from igrins.libs.master_calib import load_ref_data
        f = load_ref_data(self.caldb.helper.config, band=self.band,
                          kind=kind)
        return f

    def load_resource_for(self, resource_type):
        return self.caldb.load_resource_for(self.basename, resource_type)
