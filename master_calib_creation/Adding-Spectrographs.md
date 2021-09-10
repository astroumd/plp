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



### Identified Lines JSON

### Echellogram JSON

### Telluric Model dat

### Telluric Transmission npy

### Vega Spectrum npy

### Dead Pixel Mask FITS
