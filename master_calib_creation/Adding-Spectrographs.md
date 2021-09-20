# Adding Spectrographs

A couple steps a required to add a spectrograph to the pipeline

1) Add a folder with the spectrograph's name to the `master_calib` directory 
2) Create the needed [master calibration files](#master-calibration-files) and add them to the new directory
3) Add relevant helper functions

## Master Calibration Files

There are a number of calibrations files required for the pipeline that are spectrograph specific,
and there are some convenient functions located withing [creation.py](./creation.py) that will create these files

**Calibration Files**

* [Master Calibration config](#master-calibration-config)
* [One Dimensional Reference Spectrum JSON](#one-dimensional-reference-spectrum-json)
* [Identified Lines JSON](#identified-lines-json)
* [Reference Lines dat file](#reference-lines-dat-file)
* [Reference ohlines indices](#reference-ohlines-indices)
* [Echellogram JSON](#echellogram-json)
* [Telluric Model dat](#telluric-model-dat)
* [Telluric Transmission npy](#telluric-transmission-npy)
* [Vega Spectrum npy](#vega-spectrum-npy)
* [Dead Pixel Mask FITS](#dead-pixel-mask-fits)

### Master Calibration config

The master calibration config file provides the pipeline with information about some of the spectrographs parameters,
and the paths to the other calibration files.

#### Format

* The master calibration config file should have the name `master_cal.config`
* Plain text file
* First line of the file should be `[MASTER_CAL]`
* Parameters should be given in the format `<PARAMETER>=<VALUE>`
  * Example `GAIN_HK=2.0` or `VEGA_SPEC=A0V/vegallpr25.50000resam5.npy`

#### Parameters

1) Constant Values
    * `GAIN_<BAND>` - provide gain for each specific band
      * Example
      
            GAIN_HK=2.0
            GAIN_YJ=2.2
            
2) Paths to other calibration files from the `master_cal.config` location
    * [SOURCE TYPE]_REFSPEC_JSON
        * [SOURCE TYPE] depends on wavelength calibration recipe being used, and can be:
            * SKY
            * THAR
            * ARCS
    * [SOURCE TYPE]_IDENTIFIED_LINES_V0_JSON
    * ECHELLOGRAM_JSON
    * TELL_WVLSOL_MODEL
    * TELFIT_MODEL
    * OHLINES_INDICES_JSON
    * OHLINES_JSON
    * HITRAN_BOOTSTRAP_K
    * VEGA_SPEC
    * DEFAULT_DEADPIX_MASK

#### Example

    [MASTER_CAL]
    GAIN_H=2.0
    GAIN_K=2.2
    
    REFDATE=20140525
    
    THAR_REFSPEC_JSON=%(REFDATE)s/ThAr/SDC%(BAND)s_%(REFDATE)s_0003.oned_spec.json
    THAR_IDENTIFIED_LINES_V0_JSON=%(REFDATE)s/ThAr/ThAr_SDC%(BAND)s_%(REFDATE)s.identified_lines_v0.json
    
    SKY_REFSPEC_JSON=%(REFDATE)s/SKY/SDC%(BAND)s_%(REFDATE)s_0029.oned_spec.json
    SKY_IDENTIFIED_LINES_V0_JSON=%(REFDATE)s/SKY/SKY_SDC%(BAND)s_%(REFDATE)s.identified_lines_v0.json
    
    ECHELLOGRAM_JSON=%(REFDATE)s/SDC%(BAND)s_%(REFDATE)s.echellogram.json
    
    TELL_WVLSOL_MODEL=TelluricModel.dat
    TELFIT_MODEL=telluric/transmission-795.20-288.30-41.9-45.0-368.50-3.90-1.80-1.40.%(BAND)s.npy
    
    OHLINES_INDICES_JSON=ref_ohlines_indices_20140316.json
    OHLINES_JSON=ohlines.dat
    
    HITRAN_BOOTSTRAP_K=hitran_bootstrap_K_20140316.json
    
    VEGA_SPEC=A0V/vegallpr25.50000resam5.npy
    
    DEFAULT_DEADPIX_MASK=deadpix_mask_20140316_%(BAND)s.fits

### One Dimensional Reference Spectrum JSON

#### Format

* Two keywords:
    * "specs"
        * contains a list of lists for each spectral order
        * each spectral order list contains a list of the intensity value for each position along the order. This list should have the same length as the detector dimensions, e.g. 2048 for H2RG, and 4096 for H4RG
    * "orders"
        * List of integers containing the spectral order number. The order of this list corresponds to the order of the "specs" list

#### Example

If you have a spectrograph with 2 spectral orders, and a detector array size of 8 pixels, your one dimensional reference spectrum could look something like this:

    {
    "specs": [
        [0, 2, 3, 4, 5, 23, 495, 9483],
        [1, 42, 56, 3245, 23, 2, 404.3, 0]
    ],
    "orders": [32, 33]
    }

#### Relevant creation.py functions

1) `master_calib_creation.creation.gen_oned_spec` - generates one dimensional reference spectra, as decribed above. Function requires the following inputs:
    1) `order_map_file` - the spectral order number for each pixel in FITS format
    2) `twod_spectrum_file` - the spectrum that needs reduced
    3) `output_file` - the output JSON file path
    4) `aggregation_axis` - the numpy axis along which the spectrum should be reduced, default is 0
    5) `aggregation` - aggregation type desired for the reduction, default is nanmedian

### Reference Lines dat file

### Identified Lines JSON

Calibration file that correlates wavelength, pixel position along each order and the reference line from the lines list.

#### Format

* 5 keywords
    * "orders"
        * List of integers containing the spectral order number. The order of this list corresponds to the order of the "specs" list
    * "pixpos_list"
        * contains a list of lists for each spectral order
        * each spectral order list contains a list of the pixel position along the spectral order for each identified line
    * "ref_indices_list"
        * contains a list of lists for each spectral order
        * each spectral order list contains a list of the identified line from the "ref_name" file along the spectral order for each identified line
    * "wvl_list"
        * contains a list of lists for each spectral order
        * each spectral order list contains a list of the wavelength in microns along the spectral order for each identified line
    * "ref_name"
        * path to reference containing the lines list for the given source

#### Example

Given the following reference dat file, containing a lines list and the subsequent identified line data for each of the orders

*foo_lines.dat*

|line_number	|wvl	|intensity	|
|---	|---	|---	|
|0  	|0.95321	|1.2e-7	|
|1  	|0.96213	|1.5e-5	|
|2  	|0.97341	|1.2e-3	|
|3  	|0.98312	|9.3e-1	|
|4  	|0.99999	|4.5e-8	|
|5  	|0.10001	|3.2e-9	|

*Order 30 table*

|pixpos	|ref_indices	|wvl	|
|---	|---	|---	|
|350	|0	|0.95321	|
|450	|1	|0.96213	|
|900	|2	|0.97341	|

*Order 31 table*

|pixpos	|ref_indices	|wvl	|
|---	|---	|---	|
|425	|3	|0.98312	|
|540	|4	|0.99999	|
|542	|5	|0.10001	|

With those inputs, the output identified lines JSON file would be look like this:

    {
    "orders": [30, 31],
    "pixpos_list": [[350, 450, 900],[425, 540, 542]],
    "wvl_list": [[0.95321, 0.96213, 0.97341],[0.98312, 0.99999, 0.10001]],
    "ref_indices_list": [[0, 1, 2], [3, 4, 5]],
    "ref_name": "foo_lines.dat"
    }

#### Relevant creation.py functions

1) `master_calib_creation.creation.gen_identified_lines` - generates one dimensional reference spectra, as decribed above. Function requires the following inputs:
    1) `order_map_file` - the spectral order number for each pixel in FITS format

### Reference ohlines indices



### Echellogram JSON

### Telluric Model dat

### Telluric Transmission npy

### Vega Spectrum npy

### Dead Pixel Mask FITS

Binary pixel mask detailing which pixels are dead, and require interpolation
