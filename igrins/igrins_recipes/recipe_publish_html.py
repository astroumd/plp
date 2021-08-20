import os

from jinja2 import Environment, FileSystemLoader

from ..igrins_libs.igrins_config import IGRINSConfig
from ..igrins_libs.path_info import get_zeropadded_groupname
from ..pipeline.steps import Step
from ..utils.json_helper import json_dumps

sources = []

def publish_html(obsset):
    re = obsset.recipe_entry

    utdate = obsset.rs.basename_helper.obsdate
    band = obsset.rs.basename_helper.band

    config = IGRINSConfig()

    dirname = config.get_value("HTML_PATH", utdate)

    env = Environment(loader=FileSystemLoader('jinja_templates'))
    if obsset.expt.lower() == 'rimas':
        template = env.get_template('index_rimas.html')
    else:
        template = env.get_template('index.html')

    #sources = make_html(utdate, dirname)

    #Look is sources array to see if the other band has already been generated
    tmp = ' '
    obsids = tmp.join(re["obsids"])
    nsources = len(sources)
    has_source = False
    for i in range(nsources):
        if obsids == sources[i]["obsids"]:
            source = sources[i]
            has_source = True
            break

    if not has_source:
        source = {"name": re["objname"],
                  "obj": re["obstype"],
                  "grp1": re["group1"],
                  "grp2": re["group2"],
                  "exptime": float(re["exptime"]),
                  "recipe": re["recipe"],
                  "obsids": tmp.join(re["obsids"]),
                  "frametypes": tmp.join(re["frametypes"]),
                  "nexp": len(re["obsids"]),
                  }

        sources.append(source)

    objroot = get_zeropadded_groupname(source["grp1"])
    
    p = "igrins_spec_%s_%s.html" % (objroot, band)
    if os.path.exists(os.path.join(dirname, p)):
        source["url_%s" % band] = p

    p = "igrins_spec_%sA0V_%s.html" % (objroot, band)
    if os.path.exists(os.path.join(dirname, p)):
        print("DIRNAME, P:", dirname, p)
        print("url_%s_A0V" % band)
        source["url_%s_A0V" % band] = p
            

    json_dumps(dict(utdate=utdate,
                    sources=[source]),
               open(os.path.join(dirname, "summary.json"), "w"))

    #This will only render the obsset sent to this function
    s = template.render(utdate=utdate, sources=sources)
    open(os.path.join(dirname, "index.html"), "w").write(s)

steps = [Step("Publish HTML", publish_html),
        ]
