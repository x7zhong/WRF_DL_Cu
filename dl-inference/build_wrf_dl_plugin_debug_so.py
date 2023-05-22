import cffi
ffibuilder = cffi.FFI()

header = """
extern void infer_init_cu(int);
extern void infer_run_cu(int, int, int, int, \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*,  \
                      float*, \
                      float*, \
                      float*);
    
extern void save_fortran_array2(float*, int, int, char*);
extern void save_fortran_array3(float*, int, int, int, char*);
extern void save_fortran_array(int, int, int, int, int, int, \
                      int, int, int, int, int, int, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \
                      float*, \                
                      char*);
"""

module = """
from my_dl_plugin import ffi
import numpy as np
import datetime
import my_infer_module
from my_infer_module import OnnxEngine
import mskf_data_preprocess_short as mskfpre
import mskf_data_postprocess as mskfpost
import os,threading
import time

runner_list = []
pidname = os.getpid()
main_threadid = threading.currentThread()
logger_file_name = "./wrf_plugin_python_pid"+str(pidname)+ "_tid_"+ main_threadid.getName() +str(main_threadid.ident)+".log"
plugin_logger = my_infer_module.generate_logger(logger_file_name)
    
@ffi.def_extern()
def infer_init_cu(use_gpu=0):
    plugin_logger.info("InferInit begin")
    plugin_logger.info("ProcessID:{}, threadID:{}, threadName:{}".format(os.getpid(), threading.currentThread().ident, threading.currentThread().getName()))
    onnxrunner = OnnxEngine("./model/model_mskf.onnx", use_gpu)
    plugin_logger.info("Onnx provider :{}".format(onnxrunner.get_providers()) )
    plugin_logger.info("InferInit finished")
    runner_list.append(onnxrunner)

@ffi.def_extern()
def infer_run_cu(shape_x, shape_y, shape_z, shape_z_lev,\
          pratec, \
          nca,    \
          hfx,    \
          ust,    \
          pblh,  \
          u,      \
          v,      \
          w,      \
          t,      \
          qv,     \
          p,      \
          th,     \
          dz8w,   \
          rho,    \
          pi,     \
          w0avg,  \
          rthcuten, \
          rqvcuten, \
          rqccuten, \
          rqcruten, \
          rqicuten, \
          rqscuten):

    plugin_logger.info("Infer Run start")
    plugin_logger.info("ProcessID:{}, threadID:{}, threadName:{}".format(os.getpid(), threading.currentThread().ident, threading.currentThread().getName()))
        
    len_2d = shape_x * shape_y

    data_load_begin_time = time.time()
            
    pratec_array = my_infer_module.PtrAsarray(ffi, pratec, (shape_x, shape_y))      
    nca_array = my_infer_module.PtrAsarray(ffi, nca, (shape_x, shape_y))      
    hfx_array = my_infer_module.PtrAsarray(ffi, hfx, (shape_x, shape_y))      
    ust_array = my_infer_module.PtrAsarray(ffi, ust, (shape_x, shape_y))      
    pblh_array = my_infer_module.PtrAsarray(ffi, pblh, (shape_x, shape_y))      

    u_array = my_infer_module.PtrAsarray(ffi, u, (shape_z, shape_x, shape_y), [len_2d])      
    
    v_array = my_infer_module.PtrAsarray(ffi, v, (shape_z_lev, shape_x, shape_y), [len_2d])      
                                                      
    t_array = my_infer_module.PtrAsarray(ffi, t, (shape_z, shape_x, shape_y), [len_2d])      

    qv_array = my_infer_module.PtrAsarray(ffi, qv, (shape_z_lev, shape_x, shape_y), [len_2d])    

    p_array = my_infer_module.PtrAsarray(ffi, p, (shape_z, shape_x, shape_y), [len_2d])      
    
    th_array = my_infer_module.PtrAsarray(ffi, th, (shape_z, shape_x, shape_y), [len_2d])      
    
    dz8w_array = my_infer_module.PtrAsarray(ffi, dz8w, (shape_z, shape_x, shape_y), [len_2d])      
    
    rho_array = my_infer_module.PtrAsarray(ffi, rho, (shape_z, shape_x, shape_y), [len_2d])      
    
    pi_array = my_infer_module.PtrAsarray(ffi, pi, (shape_z, shape_x, shape_y), [len_2d])      
    
    w0avg_array = my_infer_module.PtrAsarray(ffi, w0avg, (shape_z, shape_x, shape_y), [len_2d])           

    rthcuten_array = my_infer_module.PtrAsarray(ffi, rthcuten, (shape_z, shape_x, shape_y),[len_2d])           

    rqvcuten_array = my_infer_module.PtrAsarray(ffi, rqvcuten, (shape_z, shape_x, shape_y),[len_2d])           

    rqccuten_array = my_infer_module.PtrAsarray(ffi, rqccuten, (shape_z, shape_x, shape_y),[len_2d])           
     
    rqrcuten_array = my_infer_module.PtrAsarray(ffi, rqrcuten, (shape_z, shape_x, shape_y),[len_2d])           

    rqicuten_array = my_infer_module.PtrAsarray(ffi, rqicuten, (shape_z, shape_x, shape_y),[len_2d])           

    rqscuten_array = my_infer_module.PtrAsarray(ffi, rqscuten, (shape_z, shape_x, shape_y),[len_2d])           
    
    w_array = my_infer_module.PtrAsarray(ffi, w, (shape_z_lev, shape_x, shape_y))      

    data_load_end_time = time.time()
    total_time = data_load_end_time - data_load_begin_time

    plugin_logger.info("load data takes {:.4f}s".format(total_time))

    np.savez('input.npz', pratec=pratec_array, nca=nca_array, hfx=ust_array, \
             pblh=pblh_array, u=u_array, v=v_array, \
	         w=w_array, t=t_array, qv=qv_array, p=p_array, \
	         th=th_array, dz8w=dz8w_array, rho=rho_array, pi=pi_array, w0avg=w0avg_array, \
	         rthcuten=rthcuten_array, rqvcuten=rqvcuten_array, \
             rqccuten=rqccuten_array, rqrcuten=rqrcuten_array, \
             rqicuten=rqicuten_array, rqscuten=rqscuten_array) 
        
    plugin_logger.info("Infer Run, before mskf_preprocess")
    
    data_preprocess_begin_time = time.time()
    
    feature, feature_all_variable = mskfpre.mskf_preprocess(pratec_array, nca_array, hfx_array, ust_array, \
    pblh_array, u_array, v_array, w_array, t_array, qv_array, \
    p_array, th_array, dz8w_array, rho_array, pi_array, w0avg_array, \
    rthcuten_array, rqvcuten_array, rqccuten_array, rqrcuten_array, rqicuten_array, rqscuten_array)

    data_preprocess_end_time = time.time()
    total_time = data_preprocess_end_time - data_preprocess_begin_time
    plugin_logger.info("preprocess data takes {:.4f}s".format(total_time))
    
    plugin_logger.info("Start inference")
    plugin_logger.info("feature shape is {}".format(feature.shape))
    
    inference_begin_time = time.time()
    
    onnxrunner = runner_list[0]
    predicts = []
    predicts_cf = []
    try:
        predicts_cf, predicts = onnxrunner([feature])
    except Exception as inst :
        plugin_logger.info(f"inference exception:{inst}")
    
    inference_end_time = time.time()    
    total_time = inference_end_time - inference_begin_time    
    plugin_logger.info("inference takes {:.4f}s".format(total_time))

    plugin_logger.info("Infer Run finished ")

    label_single_height_variable = ['nca', 'pratec']
    label_multi_height_variable = ['rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']

    label_all_variable_reg = label_single_height_variable + label_multi_height_variable
    
    norm_method = 'abs-max'
    
    predicts_unnorm = mskfpost.unnormalized(
    predicts, norm_mapping, label_all_variable_reg, norm_method)    
        
    plugin_logger.info("Infer Run, after mskf_post ")

    index_w0avg_output = feature_all_variable.index('w0avg_output')
    w0avg_predict = feature[:, index_w0avg_output, :].reshape((-1, ))
    
    nca_predict = predicts_unnorm[:, 1, 0].reshape((-1, ))
    pratec_predict = predicts_unnorm[:, 2, 0].reshape((-1, ))
   
    plugin_logger.info("Copy data to output variables")
    
    np.copyto(nca_array, nca_predict)
    np.copyto(pratec_array, pratec_predict)
    np.copyto(w0avg_array, w0avg_predict)
    
    rthcuten_predict = predicts_unnorm[:, 3, :].reshape((-1, ))
    rqvcuten_predict = predicts_unnorm[:, 4, :].reshape((-1, ))
    rqccuten_predict = predicts_unnorm[:, 5, :].reshape((-1, ))
    rqrcuten_predict = predicts_unnorm[:, 6, :].reshape((-1, ))
    rqicuten_predict = predicts_unnorm[:, 7, :].reshape((-1, ))
    rqscuten_predict = predicts_unnorm[:, 8, :].reshape((-1, ))
    
    np.copyto(rthcuten_array, rthcuten_predict)
    np.copyto(rqvcuten_array, rqvcuten_predict)
    np.copyto(rqccuten_array, rqccuten_predict)
    np.copyto(rqrcuten_array, rqrcuten_predict)
    np.copyto(rqicuten_array, rqicuten_predict)
    np.copyto(rqscuten_array, rqscuten_predict)

    '''
    np.savez('output.npz', pratec=pratec_array, nca=nca_array, w0avg=w0avg_array, \
	         rthcuten=rthcuten_array, rqvcuten=rqvcuten_array, \
             rqccuten=rqccuten_array, rqrcuten=rqrcuten_array, \
             rqicuten=rqicuten_array, rqscuten=rqscuten_array)            
    '''
    
        
@ffi.def_extern()
def save_fortran_array2(data_ptr, in_x, in_y, filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    filenam = filenam.replace(" ", "")

    plugin_logger.info("save to {}".format(filenam))    
    
    # n = 5
    # in_x_new = in_x - 2*n - 1
    # in_y_new = in_y - 2*n - 1
        
    # len_2d = in_x_new * in_y_new      
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y))   
    
    np.save(filenam, data_array)
    
@ffi.def_extern()
def save_fortran_array3(data_ptr, in_x, in_y, in_z, filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    filenam = filenam.replace(" ", "")    

    plugin_logger.info("save to {}".format(filenam))        
        
#    n = 5
#    in_x_new = in_x - 2*n - 1
#    in_z_new = in_z - 2*n - 1
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y, in_z)) 
        
    np.save(filenam, data_array)
    
@ffi.def_extern()
def save_fortran_array(ims, ime, jms, jme, kms, kme,
             its, ite, jts, jte, kts, kte, 
             pratec,                                                                                 &
             nca,                                                                                    &
             hfx,                                                                                    &   
             ust,                                                                                    &
             pblh,                                                                                   &   
             u,                                                                                      &              
             v,                                                                                      &
             w,                                                                                      &
             t,                                                                                      &
             qv,                                                                                     &
             p,                                                                                      &
             th,                                                                                     &
             dz8w,                                                                                   &
             rho,                                                                                    &
             pi,                                                                                     &
             w0avg,                                                                                  &
             rthcuten,                                                                               &
             rqvcuten,                                                                               &
             rqccuten,                                                                               &              
             rqrcuten,                                                                               &
             rqicuten,                                                                               &
             rqscuten,                                                                               &
             filename):   
        
    shape_x = ime-ims+1
    shape_y = jme-jms+1
    shape_z = kme-1-kms+1
    shape_z_lev = kme-kms+1
        
    filenam = ffi.string(filename).decode()
    
    #n is number of points of lateral boundary    
    n = 5
    shape_x_new = shape_x - 2*n - 1
    shape_y_new = shape_y - 2*n - 1
    
    len_2d = shape_x_new * shape_y_new
       
    plugin_logger.info("save 1")
    
    pratec_array = my_infer_module.PtrAsarray(ffi, pratec, (shape_y, shape_x), (n, len_2d))      
    nca_array = my_infer_module.PtrAsarray(ffi, nca, (shape_y, shape_x), (n, len_2d))    
    hfx_array = my_infer_module.PtrAsarray(ffi, hfx, (shape_y, shape_x), (n, len_2d))   
    ust_array = my_infer_module.PtrAsarray(ffi, ust, (shape_y, shape_x), (n, len_2d))    
    pblh_array = my_infer_module.PtrAsarray(ffi, pblh, (shape_y, shape_x), (n, len_2d))  
    pratec_out_array = my_infer_module.PtrAsarray(ffi, pratec_out, (shape_y, shape_x), (n, len_2d))  
    nca_out_array = my_infer_module.PtrAsarray(ffi, nca_out, (shape_y, shape_x), (n, len_2d))  
    
    u_array = my_infer_module.PtrAsarray(ffi, u, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    v_array = my_infer_module.PtrAsarray(ffi, v, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))
                                                      
    t_array = my_infer_module.PtrAsarray(ffi, t, (shape_y, shape_x,shape_z), \
                                            (n, len_2d, shape_z))

    qv_array = my_infer_module.PtrAsarray(ffi, qv, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))

    th_array = my_infer_module.PtrAsarray(ffi, th, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    dz8w_array = my_infer_module.PtrAsarray(ffi, dz8w, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    rho_array = my_infer_module.PtrAsarray(ffi, rho, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    pi_array = my_infer_module.PtrAsarray(ffi, pi, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    w0avg_array = my_infer_module.PtrAsarray(ffi, w0avg, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    rthcuten_array = my_infer_module.PtrAsarray(ffi, rthcuten, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqvcuten_array = my_infer_module.PtrAsarray(ffi, rqvcuten, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqcuten_array = my_infer_module.PtrAsarray(ffi, rqccuten, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
        
    rqrcuten_array = my_infer_module.PtrAsarray(ffi, rqrcuten, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqicuten_array = my_infer_module.PtrAsarray(ffi, rqicuten, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqscuten_array = my_infer_module.PtrAsarray(ffi, rqscuten, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))            
                     
    w_array = my_infer_module.PtrAsarray(ffi, w, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))    
        
    np.savez(filenam, pratec=pratec_array, nca=nca_array, hfx=hfx_array, \
             ust=ust_array, pblh=pblh_array, u=u_array, \
	         v=v_array, w=w_array, t=t_array, qv=qv_array, \
	         p=p_array, th=th_array, dz8w=dz8w_array, rho=rho_array, pi=pi_array, \
	         qi=qi_array, qs=qs_array, qg=qg_array, o3vmr=o3vmr_array, cldfrac=cldfrac_array, \
             w0avg=w0avg_array, rthcuten=rthcuten_array, rqvcuten=rqvcuten_array, \
             rqccuten=rqccuten_array, rqrcuten=rqrcuten_array, rqicuten=rqicuten_array,\
             rqscuten=rqscuten_array, \
             its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, \
             ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme)
    
"""

with open("my_dl_plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_dl_plugin", r'''
    #include "my_dl_plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libDL_inference_plugin.so", verbose=True)
