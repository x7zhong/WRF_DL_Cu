# Code description:
## WRF
Contains the files that are different from the original WRF v4.3 files and used for implementing the WRF-ML coupler.
To run the WRF coupled with ML-based mskf cumulus schemes, you need to add files that did not exist or overwrite the existing WRF files.

### phys
module_cu_mskf_dl_replace.F: 
subroutine infer_init_cu: used to initialize the ML-based mskf cumulus scheme, this is called in main/wrf.F
subrountine infer_run_cu: used to run the the ML-based mskf cumulus scheme
subroutine save_fortran_array: used to save wrf variables into npy or npz files

module_cu_mskf.F: difference from the original module_cu_mskf.F is that it outputs one 2 dimensional variable pblh, one 3 dimensional layer variable u, and one 3 dimensional level variable w, Yu Xing will help to check whether these variables are consistent with the npz files that he provided before using this method.

module_cumulus_driver.F:  difference from the original module_cumulus_driver.F is that the same as changes in module_cu_mskf.F

Makefile: used to add module_cu_mskf_dl_replace.o

### main 
wrf.F: call infer_init_cu to initialize ML models

depend.common: add dependencies for new files such as module_cu_mskf_dl_replace.o

## dl_inference
Contains the files used for building the library in dl-inference-plugin, and the python script to run ML models within WRF_DL

build_wrf_dl_plugin_debug_so.py: build python functions and generate the library in dl-inference-plugin
