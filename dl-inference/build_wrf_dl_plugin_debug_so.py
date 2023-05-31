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

extern void save_xland(float*, int,int,int, int, char*);
extern void save_horizontal(float*, int,int,char*);
extern void save_fortran_array2(float*, int, int, char*);
extern void save_fortran_array3(float*, int, int, int, char*);
extern void save_fortran_array(int, int, int, int, int, int, int, int, int, int, int, int,\
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
from config_wrf import config_wrf
from config_norm import norm_mapping
from cu_utils import mskf_constraint, trigger_consistency, nca_constraint

runner_list = []
pidname = os.getpid()
main_threadid = threading.currentThread()
logger_file_name = "./wrf_plugin_python_pid"+str(pidname)+ "_tid_"+ main_threadid.getName() +str(main_threadid.ident)+".log"
plugin_logger = my_infer_module.generate_logger(logger_file_name)
    
@ffi.def_extern()
def infer_init_cu(use_gpu=0):
    plugin_logger.info("InferInit begin")
    plugin_logger.info("ProcessID:{}, threadID:{}, threadName:{}".format(os.getpid(), threading.currentThread().ident, threading.currentThread().getName()))
    plugin_logger.info(os.path.isfile("./model/model_mskf.onnx"))
    # onnxrunner = OnnxEngine("./model/model_mskf.onnx", use_gpu)
    onnxrunner = OnnxEngine("./model/model_mskf.onnx", -1)
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
          rqrcuten, \
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

    # plugin_logger.info("nca_array max {}, min {}".format(nca_array.max(), nca_array.min()))

    # plugin_logger.info("pblh_array max {}, min {}".format(pblh_array.max(), pblh_array.min()))

    u_array = my_infer_module.PtrAsarray(ffi, u, (shape_z, shape_x, shape_y), [len_2d])      
    
    # plugin_logger.info("pblh_array max {}, min {}".format(pblh_array.max(), pblh_array.min()))
    
    v_array = my_infer_module.PtrAsarray(ffi, v, (shape_z, shape_x, shape_y), [len_2d])      
                                                      
    t_array = my_infer_module.PtrAsarray(ffi, t, (shape_z, shape_x, shape_y), [len_2d])      

    qv_array = my_infer_module.PtrAsarray(ffi, qv, (shape_z, shape_x, shape_y), [len_2d])    

    p_array = my_infer_module.PtrAsarray(ffi, p, (shape_z, shape_x, shape_y), [len_2d])      
    
    # plugin_logger.info("p_array max {}, min {}".format(p_array.max(), p_array.min()))
    
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
    
    w_array = my_infer_module.PtrAsarray(ffi, w, (shape_z_lev, shape_x, shape_y),[len_2d])           

    w0avg_out_array = my_infer_module.PtrAsarray(ffi, w0avg, (shape_z, shape_x, shape_y))           

    rthcuten_out_array = my_infer_module.PtrAsarray(ffi, rthcuten, (shape_z, shape_x, shape_y))           

    rqvcuten_out_array = my_infer_module.PtrAsarray(ffi, rqvcuten, (shape_z, shape_x, shape_y))           

    rqccuten_out_array = my_infer_module.PtrAsarray(ffi, rqccuten, (shape_z, shape_x, shape_y))           
     
    rqrcuten_out_array = my_infer_module.PtrAsarray(ffi, rqrcuten, (shape_z, shape_x, shape_y))           

    rqicuten_out_array = my_infer_module.PtrAsarray(ffi, rqicuten, (shape_z, shape_x, shape_y))           

    rqscuten_out_array = my_infer_module.PtrAsarray(ffi, rqscuten, (shape_z, shape_x, shape_y))           
        
#     np.savez('infer.npz', pratec=pratec_array, nca=nca_array, hfx=hfx_array, \
#              ust=ust_array, pblh=pblh_array, u=u_array, \
# 	         v=v_array, w=w_array, t=t_array, q=qv_array, \
# 	         p=p_array, th=th_array, dz8w=dz8w_array, rho=rho_array, pi=pi_array, \
#              w0avg=w0avg_array, rthcuten=rthcuten_array, rqvcuten=rqvcuten_array, \
#              rqccuten=rqccuten_array, rqrcuten=rqrcuten_array, rqicuten=rqicuten_array,\
#              rqscuten=rqscuten_array)
             
    data_load_end_time = time.time()
    total_time = data_load_end_time - data_load_begin_time

    # plugin_logger.info("load data takes {:.4f}s".format(total_time))
    
    plugin_logger.info("Infer Run, before mskf_preprocess")
    
    data_preprocess_begin_time = time.time()
    
    feature, feature_all_variable = mskfpre.mskf_preprocess(pratec_array, nca_array, \
     hfx_array, ust_array, pblh_array, u_array, v_array, w_array, \
     t_array, qv_array, p_array, th_array, dz8w_array, rho_array, pi_array, w0avg_array,  \
      rthcuten_array, rqvcuten_array, rqccuten_array, rqrcuten_array,rqicuten_array,rqscuten_array)

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
    
    plugin_logger.info("Infer Run, predicts_cf shape {}, predicts shape {}".format(predicts_cf.shape, predicts.shape))
    
    inference_end_time = time.time()    
    total_time = inference_end_time - inference_begin_time    
    plugin_logger.info("inference takes {:.4f}s".format(total_time))

    plugin_logger.info("Infer Run finished ")

    label_all_variable_cf = ['trigger']

    label_single_height_variable = ['nca', 'pratec']
    label_multi_height_variable = ['rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']

    label_all_variable_reg = label_single_height_variable + label_multi_height_variable
            
    # plugin_logger.info("mskf_constraint")
    
    ListofVar = config_wrf.label_all_variable_reg
    predicts = mskf_constraint(feature, predicts, ListofVar, config_wrf)
    
    ListofVar = ['trigger']  
    predicts_cf = mskf_constraint(feature, predicts_cf, ListofVar, config_wrf)
    
    # plugin_logger.info("trigger_consistency")
    
    predicts = trigger_consistency(predicts, predicts_cf, config_wrf)
    
    plugin_logger.info("Start mskf_post unnormalized")
    
    predicts_unnorm = mskfpost.unnormalized(
    predicts, norm_mapping, label_all_variable_reg, config_wrf.norm_method)   
    
    #Ensure nca to be integer
    predicts_unnorm = nca_constraint(predicts_unnorm, config_wrf.label_all_variable_reg)            

    plugin_logger.info("Infer Run, after mskf_post, predicts_unnorm shape {}".format(predicts_unnorm.shape))

    index_w0avg_output = feature_all_variable.index('w0avg_output')
    w0avg_predict = feature[:, index_w0avg_output, :].reshape((-1, ))
    #Unnormalize w0avg_predict
    variable_name = 'w0avg'
    if config_wrf.norm_method == 'z-score':                                
        
        w0avg_predict = w0avg_predict * \
        norm_mapping[variable_name]["scale"] + \
        norm_mapping[variable_name]["mean"]
        
    elif config_wrf.norm_method == 'min-max':                                
        w0avg_predict = w0avg_predict * \
        norm_mapping[variable_name]["max"]

    elif config_wrf.norm_method == 'abs-max':   
        w0avg_predict = w0avg_predict * \
        max( abs(norm_mapping[variable_name]['max']), \
        abs(norm_mapping[variable_name]['min']) )    
    
    plugin_logger.info("w0avg_predict max {}, min {}".format(w0avg_predict.max(), w0avg_predict.min()))
    
    nca_predict = predicts_unnorm[:, 0, 0].reshape((-1, ))
    #nca needs to be multiplied by dt
    nca_predict = nca_predict * config_wrf.dt
    
    pratec_predict = predicts_unnorm[:, 1, 0].reshape((-1, ))
    
    plugin_logger.info("Copy data to output variables")

    np.copyto(w0avg_out_array, w0avg_predict)            

    np.copyto(nca_array, nca_predict)            
    np.copyto(pratec_array, pratec_predict)

    plugin_logger.info("Before normalization nca max {}, min {}".format(\
    predicts[:, 0, 0].max(), predicts[:, 0, 0].min()))    
    
    plugin_logger.info("Shape of nca_predict is {}, max {}, min {}".format(\
    nca_predict.shape, nca_predict.max(), nca_predict.min()))    
    
    plugin_logger.info("Shape of pratec_predict is {}, max {}, min {}".format(\
    pratec_predict.shape, pratec_predict.max(), pratec_predict.min()))    
                          
    rthcuten_predict = predicts_unnorm[:, 2, :].reshape((-1, ))    
    rqvcuten_predict = predicts_unnorm[:, 3, :].reshape((-1, ))    
    rqccuten_predict = predicts_unnorm[:, 4, :].reshape((-1, ))    
    rqrcuten_predict = predicts_unnorm[:, 5, :].reshape((-1, ))
    rqicuten_predict = predicts_unnorm[:, 6, :].reshape((-1, ))    
    rqscuten_predict = predicts_unnorm[:, 7, :].reshape((-1, ))
        
    plugin_logger.info("Shape of rthcuten_predict is {}, max {}, min {}".format(\
    rthcuten_predict.shape, rthcuten_predict.max(), rthcuten_predict.min()))
        
    np.copyto(rthcuten_out_array, rthcuten_predict)
    np.copyto(rqvcuten_out_array, rqvcuten_predict)
    np.copyto(rqccuten_out_array, rqccuten_predict)
    np.copyto(rqrcuten_out_array, rqrcuten_predict)
    np.copyto(rqicuten_out_array, rqicuten_predict)
    np.copyto(rqscuten_out_array, rqscuten_predict)
    
    # np.savez('output.npz', nca=nca_array, pratec=pratec_array, \
    #          w0avg=w0avg_out_array, rthcuten=rthcuten_out_array, rqvcuten=rqvcuten_out_array, \
    #          rqccuten=rqccuten_out_array, rqrcuten=rqrcuten_out_array, rqicuten=rqicuten_out_array,\
    #          rqscuten=rqscuten_out_array)
        
    plugin_logger.info("End Copy data")
    
@ffi.def_extern()
def save_xland(xland,nx,ny,nxy,nz,filename):
    filenam=ffi.string(filename).decode()
    xland_array = my_infer_module.PtrAsarray(ffi,xland,(nxy,1))
    print("in saving xland")
    np.savez(filenam,xland=xland_array,nx=nx,ny=ny,nxy=nxy)

@ffi.def_extern()
def save_horizontal(var,nx,ny,filename):

    filenam=ffi.string(filename).decode()
    var_array = my_infer_module.PtrAsarray(ffi,var,(nx,ny))
    np.savez(filenam,var=var_array,nx=nx,ny=ny)
        
@ffi.def_extern()
def save_fortran_array2(data_ptr, in_x, in_y, filename):    
    filenam = ffi.string(filename).decode()
    
    n = 5
    in_x_new = in_x - 2*n - 1
    in_y_new = in_y - 2*n - 1
        
    len_2d = in_x_new * in_y_new      
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_y, in_x), (n, len_2d))   
#   np.savez(filenam,var=data_array,nx=in_x_new,ny=in_y_new)
#    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y))   
    np.savez(filenam,var=data_array,nx=in_x,ny=in_y)

    
@ffi.def_extern()
def save_fortran_array3(data_ptr, in_x, in_y, in_z, filename):    
    filenam = ffi.string(filename).decode()
    
    n = 5
    in_x_new = in_x - 2*n - 1
    in_y_new = in_y - 2*n - 1
        
    len_2d = in_x_new * in_y_new      
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_y, in_x,in_z),(n,len_2d,in_z)) 
    np.savez(filenam,var=data_array,nx=in_x,ny=in_y,nz=in_z)
#   data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y,in_z)) ,(n,len_2d,in_z))
#   np.savez(filenam,var=data_array,nx=in_x_new,ny=in_y_new,nz=in_z)

@ffi.def_extern()
def save_fortran_array(ims, ime, jms, jme, kms, kme,its, ite, jts, jte, kts, kte,\
        pratec,\
     nca,    \
     hfx,    \
     ust,    \
     pblh,   \
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
     rqrcuten, \
     rqicuten, \
     rqscuten, \
     nca_out,  \
     pratec_out, \
     rthcuten_out, \
     rqvcuten_out, \
     rqccuten_out, \
     rqrcuten_out, \
     rqicuten_out, \
     rqscuten_out, \
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
    
    v_array = my_infer_module.PtrAsarray(ffi, v, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
                                                      
    t_array = my_infer_module.PtrAsarray(ffi, t, (shape_y, shape_x,shape_z), \
                                            (n, len_2d, shape_z))

    qv_array = my_infer_module.PtrAsarray(ffi, qv, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

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
