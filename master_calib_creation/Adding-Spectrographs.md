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

`master_calib_creation.creation.gen_oned_spec`
        
    Creates a one dimensional spectrum for each order, and saves in json format to the output_file location.

    Parameters
    ----------
    order_map_file : str, path
        Path to fits file containing order map
    twod_spectrum_file : str, path
        Path to fits file containing 2D spectrum, should be aligned with the order map
    output_file : str, path
        Path to output json file
    aggregation_axis : integer, 0 or 1, optional
        numpy array axis to aggregate along
    aggregation : function, optional
        Aggregation function to convert 2D to 1D. np.nanmedian, np.nanmode, np.nanmean, np.nanmax, etc

### Reference Lines dat file

Space separated reference file containing 2 columns, wavelength (angstrom) and intensity (arbitrary units)


#### Example

    6138.239 1.258e-01
    6138.265 1.258e-01
    6138.466 3.361e-01
    6138.502 3.361e-01
    6139.309 6.483e-02
    6139.357 6.483e-02
    6140.558 3.457e-02
    6140.562 3.457e-02
    6140.666 1.545e-01
    6140.702 1.545e-01
    6140.892 1.950e-02
    6140.944 1.950e-02
    6141.146 6.512e-01
    6141.184 6.512e-01
    6145.068 2.726e-01
    6145.081 2.726e-01
    6145.347 4.521e-03
    6145.392 4.521e-03
    6145.467 7.466e-03

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
|0  	|9532.1	|1.2e-7	|
|1  	|9621.3	|1.5e-5	|
|2  	|9734.1	|1.2e-3	|
|3  	|9831.2	|9.3e-1	|
|4  	|9999.9	|4.5e-8	|
|5  	|10000.1	|3.2e-9	|

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

`master_calib_creation.creation.gen_identified_lines` 

    Creates identified lines calibration file

    Parameters
    ----------
    oned_spec_json_file : str, path
        1D spectrum file containing peaks to be identified, typically generated by gen_oned_spec
    oned_wavemap_file : str, path
        1D spectrum file containing wavelength corresponding to pixel location, typically generated by gen_oned_spec
        from 2D wavemap file
    lines_dat_file : str, path
        lines list file corresponding to oned_spec_json_file spectrum
    output_file : str, path
        Path to output json file

`master_calib_creation.creation.gen_ref_indices` - Primarily for the creating reference indices json file, but also removes repetition of lines from the identified lines file.

    Generates the reference indices json calibration file

    Parameters
    ----------
    identified_lines_json_file : str, path
        identified lines json file generated by gen_identified_lines
    lines_dat_file : str, path
        lines dat file used to create identified_lines_json_file
    band_name : str
        band calibtration is being generated for
    updated_identified_lines_output : str, path
        output identified lines json file name with line degeneracies fixed
    ref_indices_output : str, path
        output reference indices json file name

### Reference ohlines indices

Calibration file that gives the line indices in each order for each band.

#### Format

* A keyword for each band, e.g. "HK", "YJ" for RIMAS or "H", "K" for IGRINS
    * Each band contains a keyword for each order, e.g. "30", "31", ..., "44" for RIMAS "YJ" band
        * Each order contains a list of lists of integers referencing the lines within the order. Line indices are determined from the reference lines data file. Blended lines are within the same list.

#### Example

    {
    "H": {
        "99": [[4060, 4061], [4074, 4075], [4076, 4077]],
        "100": [[4034, 4035], [4048, 4049]]
        }
    "K": {
        "88": [[4406, 4407], [4412, 4413], [4420], [4421], [4424, 4425], [4426], [4427]],
        "89": [[4375, 4377], [4378, 4379], [4386, 4387], [4392], [4393]]
    }

#### Relevant creation.py functions

`master_calib_creation.creation.gen_ref_indices`

    Generates the reference indices json calibration file

    Parameters
    ----------
    identified_lines_json_file : str, path
        identified lines json file generated by gen_identified_lines
    lines_dat_file : str, path
        lines dat file used to create identified_lines_json_file
    band_name : str
        band calibtration is being generated for
    updated_identified_lines_output : str, path
        output identified lines json file name with line degeneracies fixed
    ref_indices_output : str, path
        output reference indices json file name

### Echellogram JSON

Calibration file that gives the first guess for the wavelength solution for each band.

#### Format

* 4 keywords
    * "orders" - List of integers corresponding to the spectral orders within the band
    * "wvl_list" - List of lists for each order corresponding to wavelength (microns) for each x, y coordinate
    * "x_list" - List of lists for each order corresponding to the x coordinate for wavelength in "wvl_list"
    * "y_list" - List of lists for each order corresponding to the x coordinate for wavelength in "wvl_list"

#### Example

If we have an 8x8 detector array with 2 spectral orders, the resulting echellogram file would look something like this:

    {
    "orders": [30, 31],
    "wvl_list": [
            [1.123, 1.124, 1.125, 1.126, 1.127, 1.128, 1.129, 1.130],
            [1.131, 1.132, 1.133, 1.134, 1.135, 1.136, 1.137, 1.138]
        ],
    "x_list": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7]
        ],
    "y_list": [
            [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
            [5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
        ]
    }

#### Relevant creation.py functions

`master_calib_creation.creation.gen_echellogram`

    Creates echellogram calibration json file using 1D wavemap file and 2D order_map

    Parameters
    ----------
    order_map_file : str, path
        Path to fits file containing order map
    oned_wavemap_file : str, path
        1D spectrum file containing wavelength corresponding to pixel location, typically generated by gen_oned_spec
        from 2D wavemap file
    output_file : str, path
        Path to output json file
    aggregation_axis : integer, 0 or 1, optional
        numpy array axis to aggregate along
    aggregation : function, optional
        Aggregation function to convert 2D to 1D. np.nanmedian, np.nanmode, np.nanmean, np.nanmax, etc

`master_calib_creation.creation.gen_echellogram_fit_wvlsol`

    Creates echellogram calibration json file using wavelength solution

    Parameters
    ----------
    echellogram_json_file : str, path
        Echellogram file from gen_echellogram, used for 'y_list' values only
    identified_lines_json_file : str, path
        identified lines calibration file from gen_identified_lines
    ref_indices_json_file : str, path
        reference indices calibration file from gen_reference_indices
    output_file : str, path
        Path to output json file
    pixels_in_order : int
        pixels along order, 2048 for H2RG and 4096 for H4RG
    centroid_solutions_json_file : str, path, optional
        *.centroid_solutions.json file created from pipeline flat step
    domain_starting_index : int, optional
        starting domain value from centroid solutions file, 3 to skip the pipeline generated extra
    fit_output_file : str, path, optional
        json file detailing the fit parameters used for the echellogram
    pixel_degree : int, optional
        fitting degrees for pixels
    order_degree : int, optional
        fitting degrees for orders
    p_init_pickle : str, path, optional
        optional fitting pickle file to use as initial guess for echellogram fitting, must be
        of type astropy.modeling.polynomial.PolynomialBase
    pickle_output_file : str, path, optional
        output file for wavelength solution polynomial
    band : str, optional
        band from ref_indices json file

### Telluric Model dat

Space separated reference file containing 2 columns, wavelength (angstrom) and the corresponding transmission constant between 0 and 1.

#### Example

    2.599846455736084863e+03 1.068276781330053926e-17
    2.599849455736084565e+03 2.407407764009599432e-17
    2.599852455736084721e+03 2.674899896227994141e-17
    2.599855455736084878e+03 1.337451435177022928e-17
    2.599858455736084579e+03 0.000000000000000000e+00
    2.599861455736084281e+03 0.000000000000000000e+00
    2.599864455736084437e+03 0.000000000000000000e+00
    2.599867455736084594e+03 2.674916554461214267e-18
    2.599870455736084295e+03 8.024754997299774826e-18
    2.599873455736083997e+03 2.942413438372233685e-17
    2.599876455736084154e+03 1.337463036894137730e-17
    2.599879455736084310e+03 0.000000000000000000e+00
    2.599882455736084012e+03 0.000000000000000000e+00
    2.599885455736083713e+03 0.000000000000000000e+00
    2.599888455736083870e+03 0.000000000000000000e+00
    2.599891455736084026e+03 1.162997493549798436e-20
    2.599894455736083728e+03 1.113014381139330527e-17
    2.599897455736083430e+03 2.755691414632561846e-18

### Telluric Transmission npy

Telluric transmission file for each band saved as a numpy array (using `numpy.save` function) with 2 columns, wavelenth (nm) and the corresponding transmission constant between 0 and 1.

#### Example

    array([[1.30002e+03, 9.98184e-01],
       [1.30004e+03, 9.97771e-01],
       [1.30006e+03, 9.98214e-01],
       ...,
       [2.59994e+03, 3.89616e-03],
       [2.59996e+03, 7.87336e-03],
       [2.59998e+03, 1.34491e-02]])

### Vega Spectrum npy

Vega spectrum file saved as a numpy array (using `numpy.save` function) with at least 2 columns, wavelenth (angstrom) and the corresponding flux. This is a standard spectrum, so it can be copied over from RIMAS or IGRINS

#### Example

    array([[9.00005247e+01, 1.53000000e-01, 2.11700000e-01],
       [9.00014247e+01, 2.03700000e-01, 2.11800000e-01],
       [9.00023247e+01, 2.11500000e-01, 2.11800000e-01],
       ...,
       [2.99999891e+05, 1.25500000e-03, 1.68600000e-03],
       [3.00002891e+05, 1.13100000e-03, 1.68600000e-03],
       [3.00005891e+05, 9.59400000e-04, 1.68600000e-03]])

### Dead Pixel Mask FITS

Binary pixel mask detailing which pixels are dead, and require interpolation
