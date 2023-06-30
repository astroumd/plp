[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18579.svg)](http://dx.doi.org/10.5281/zenodo.18579) - publication on original IGRINS pipeline

# RIMAS Pipeline Package

The RIMAS pipeline package is a fork of the version 2 of the [IGRINS pipeline](https://github.com/igrins/plp) and the information about that pipeline in their README and [Wiki](https://github.com/igrins/plp/wiki) should be mostly applicable to the RIMAS pipeline. In addition, we have added information about the pipeline to our [Wiki](https://github.com/njmiller/plp/wiki).

The same process to run the IGRINS pipeline should be used to run the RIMAS pipeline. The only real difference when running the code is that `a0v-ab-pypeit` and `a0v-onoff-pypeit` should be run on the A0V stars to use the Pypeit telluric correction code. We found that the telluric correction provided by IGRINS did not work well for the resolution of RIMAS and as such we wrote an A0V analysis step that uses the Pypeit telluric correction code. In order to use that code, PypeIt must be installed. Installation instructions for Pypeit can be seen at https://pypeit.readthedocs.io/en/release/. An overview of the process to run the pipeline can be seen [here](https://github.com/njmiller/plp/wiki/How-to-run-pipeline)

Contact:
Nathan Miller - nathan.j.miller@nasa.gov
Joseph Durbak - jmdurbak@terpmail.umd.edu

or you can ask any questions here on Github
