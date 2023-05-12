import cffi
ffibuilder = cffi.FFI()

header = """
extern void infer_init_cu(int);
extern void infer_run_cu(int, int, int, int,     
                      float*,                      
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*,
                      float*,
                      float*);

extern void save_fortran_array2(float*, int, int, char*);
extern void save_fortran_array3(float*, int, int, int, char*);
extern void save_fortran_array(int, int, int, int, int, int,   
                      int, int, int, int, int, int,   
                      float*, 
                      float*,                      
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*,                       
                      float*, 
                      float*, 
                      float*,
                      float*,
                      float*,
                      float*,
                      float*,                      
                      char*);
"""

module = """
from my_dl_plugin import ffi
import numpy as np
import datetime
import my_infer_module
from my_infer_module import OnnxEngine
#import mskf_data_preprocess_short as mskfpre
#import mskf_data_postprocess as mskfpost
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
    #onnxrunner = OnnxEngine("./model/cu_v01.onnx", use_gpu)
    #plugin_logger.info("Onnx provider :{}".format(onnxrunner.get_providers()) )
    plugin_logger.info("InferInit finished")
    #runner_list.append(onnxrunner)

@ffi.def_extern()
def infer_run_cu(shape_x, shape_y, shape_z, shape_z_lev,    
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
          rqcruten,                                                                               &
          rqicuten,                                                                               &
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
    
    plugin_logger.info("Infer Run, before mskf_preprocess")
    
    data_preprocess_begin_time = time.time()
    
#     feature, auxiliary_feature, coszen_array = mskfpre.mskf_preprocess(mskf_array, _array, \
#     landfrac_array, icefrac_array, snow_array, solcon_array, tsfc_array, emis_array, \
#     play_array, plev_array, tlay_array, tlev_array, cldfrac_array, o3vmr_array, \
#     qc_array, qg_array, qr_array, qi_array, qs_array, qv_array)

#     data_preprocess_end_time = time.time()
#     total_time = data_preprocess_end_time - data_preprocess_begin_time
#     plugin_logger.info("preprocess data takes {:.4f}s".format(total_time))
    
#     plugin_logger.info("Start inference")
#     plugin_logger.info("feature shape is {}".format(feature.shape))
    
#     inference_begin_time = time.time()
    
#     onnxrunner = runner_list[0]
#     output =None
#     try:
#       output = onnxrunner([feature])
#     except Exception as inst :
#         plugin_logger.info(f"inference exception:{inst}")

    
#     inference_end_time = time.time()    
#     total_time = inference_end_time - inference_begin_time    
#     #plugin_logger.info("inference takes {:.4f}s".format(total_time))

#     plugin_logger.info("Infer Run finished ")
    
#     get_hr_begin_time = time.time()    
#     swhr_res, lwhr_res, swuflx_predict, swdflx_predict, lwuflx_predict, lwdflx_predict\
#     = mskfpost.rrtmg_get_hr(output, auxiliary_feature, coszen_array)
    
#     get_hr_end_time = time.time()    
#     total_time = get_hr_end_time - get_hr_begin_time    
#     #plugin_logger.info("get_hr takes {:.4f}s".format(total_time))
    
#     #plugin_logger.info("Infer Run, after mskf_post ")

#     swhr_res = swhr_res.reshape((len_2d, shape_z))
#     lwhr_res = lwhr_res.reshape((len_2d, shape_z))
#     swuflx_predict = swuflx_predict.reshape((-1, ))
#     swdflx_predict = swdflx_predict.reshape((-1, ))
#     lwuflx_predict = lwuflx_predict.reshape((-1, ))
#     lwdflx_predict = lwdflx_predict.reshape((-1, ))
    
#     np.copyto(swuflx_array, swuflx_predict)
#     np.copyto(swdflx_array, swdflx_predict)
#     np.copyto(lwuflx_array, lwuflx_predict)
#     np.copyto(lwdflx_array, lwdflx_predict)

#     #Convert sw and sw heating rate to wrf required tendency
#     rthratensw_tmp = np.zeros((len_2d, shape_z_lev))
#     rthratenlw_tmp = np.zeros((len_2d, shape_z_lev))    
#     rthratensw_tmp[:, 0:-1] = swhr_res/(pi_array[:, 0:-1]*86400)  
#     rthratenlw_tmp[:, 0:-1] = lwhr_res/(pi_array[:, 0:-1]*86400) 
#     rthraten_tmp = rthratensw_tmp + rthratenlw_tmp
        
#     rthraten_tmp = rthraten_tmp.reshape((-1,))
#     rthratensw_tmp = rthratensw_tmp.reshape((-1,))
#     rthratenlw_tmp = rthratenlw_tmp.reshape((-1,))
    
#     np.copyto(rthraten_array, rthraten_tmp)
#     np.copyto(rthratensw_array, rthratensw_tmp)
#     np.copyto(rthratenlw_array, rthratenlw_tmp)   

    #'''
    np.savez('infer.npz', pratec=pratec_array, nca=nca_array, hfx=ust_array, \
             pblh=pblh_array, u=u_array, v=v_array, \
	         w=w_array, t=t_array, qv=qv_array, p=p_array, \
	         th=th_array, dz8w=dz8w_array, rho=rho_array, pi=pi_array, w0avg=w0avg_array, \
	         rthcuten=rthcuten_array, rqvcuten=rqvcuten_array, \
             rqccuten=rqccuten_array, rqrcuten=rqrcuten_array, \
             rqicuten=rqicuten_array, rqscuten=rqscuten_array,)            
    #'''
    
        
@ffi.def_extern()
def save_fortran_array2(data_ptr, in_x, in_y, filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    filenam = filenam.replace(" ", "")

#    plugin_logger.info("save to {}".format(filenam))    
    
    n = 5
    in_x_new = in_x - 2*n - 1
    in_y_new = in_y - 2*n - 1
        
    len_2d = in_x_new * in_y_new      
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y), (n, len_2d))   
    
    np.save(filenam, data_array)
    
@ffi.def_extern()
def save_fortran_array3(data_ptr, in_x, in_y, in_z, filename):    
#    plugin_logger.info("save to {}".format(filenam))        

    filenam = ffi.string(filename).decode('UTF-8')
    filenam = filenam.replace(" ", "")    
        
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
             nca_out,                                                                                &
             pratec_out,                                                                             &
             rthcuten_out,                                                                           &
             rqvcuten_out,                                                                           &
             rqccuten_out,                                                                           &              
             rqrcuten_out,                                                                           &
             rqicuten_out,                                                                           &
             rqscuten_out
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
             
    rthcuten_out_array = my_infer_module.PtrAsarray(ffi, rthcuten_out, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqvcuten_out_array = my_infer_module.PtrAsarray(ffi, rqvcuten_out, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqcuten_out_array = my_infer_module.PtrAsarray(ffi, rqccuten_out, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
        
    rqrcuten_out_array = my_infer_module.PtrAsarray(ffi, rqrcuten_out, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqicuten_out_array = my_infer_module.PtrAsarray(ffi, rqicuten_out, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    rqscuten_out_array = my_infer_module.PtrAsarray(ffi, rqscuten_out, (shape_y, shape_x, shape_z), \
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
             rqscuten=rqscuten_array, nca_out=nca_out_array, \
             pratec_out=pratec_out_array, rthcuten_out=rthcuten_out_array, \
             rqvcuten_out=rqvcuten_out_array, rqccuten_out=rqccuten_out_array, \
             rqrcuten_out=rqrcuten_out_array, rqicuten_out=rqicuten_out_array,\
             rqscuten_out=rqscuten_out_array, \
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
