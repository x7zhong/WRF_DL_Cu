#prepare rrtmg DL model required inputs

import logging
import numpy as np
from config_wrf import config_wrf
from cu_utils import calculate_qv_saturated, calculate_rh
from config_norm import norm_mapping
    
def generate_logger(log_file_path):
    #Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Create handler for logging data to a file
    logger_handler = logging.FileHandler(log_file_path, mode = 'a')
    logger_handler.setLevel(logging.INFO)

    #Define format for handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logger_handler.setFormatter(formatter)

    #Add logger in the handler
    logger.addHandler(logger_handler)

    return logger

#nrows: number of rows
#ncols: number of cols
#batch_size = nrows * ncols   
#nlayers: number of vertical layers

#1 value: solcon
#nrows * ncols: pratec, nca, hfx, ust, pblh
#nrows * ncols * nlayers: u, v, \
#t, qv, p, th, dz8w, rho, pi, w0avg, \
#rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten
#nrows * ncols * (nlayers+1): w     
    
def mskf_preprocess(pratec, nca, hfx, ust, pblh, u, v, w, \
                    t, qv, p, th, dz8w, rho, pi, w0avg, \
                    rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten):
    
    logger = generate_logger("./mskf_preprocess.log")

    logger.info("Start mskf_preprocess")
    
    batch_size, nlayers = u.shape
    
#==============================================================================
# WRF data preprocessing
#==============================================================================
    
    logger.info("WRF data preprocess")

#==============================================================================
# Initialize variables used to input to DL model
#==============================================================================
            
#    logger.info("Initialize variables used to input to DL mskf model")

    ### single height feature ###        
    single_height_variable = ['pratec', 'nca', 'hfx', 'ust', 'pblh']
    single_feature = np.zeros([batch_size, len(single_height_variable), 1])
    
    nca = nca/config_wrf.dt 
    nca[nca < 0.01] = 0
    single_feature[:, 0, 0] = pratec
    single_feature[:, 1, 0] = nca
    single_feature[:, 2, 0] = hfx    
    single_feature[:, 3, 0] = ust    
    single_feature[:, 4, 0] = pblh
        
    ### multi height/layer feature ###            
    multi_height_variable = ['u', 'v', 'w', 't', 'q', 'p', 'th', 'dz8w', \
                             'rho', 'pi', 'w0avg', \
                             'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', \
                             'rqicuten', 'rqscuten']
        
    multi_feature = np.zeros([batch_size, len(multi_height_variable), nlayers])

    logger.info("Specify 3d variables used to input to DL mskf model")

    logger.info("Shape of multi_feature is {}".format(multi_feature.shape))

    logger.info("yux 1")
    logger.info("Shape of of u {}".format(u.shape))
    multi_feature[:, 0, :] = u
    logger.info("yux 2")
    logger.info("Shape of of v {}".format(v.shape))
    multi_feature[:, 1, :] = v
    logger.info("yux 3")

    logger.info("Shape of of w {}".format(w.shape))
    w0 = 0.5*(w[:, 0:-1] + w[:, 1:])    
    logger.info("Shape of of w0 {}".format(w0.shape))
    multi_feature[:, 2, :] = w0
    logger.info("yux 4")
    
    multi_feature[:, 3, :] = t
    multi_feature[:, 4, :] = qv
    
    #pressure here is level pressure, convert to layer pressure                    
    plev = np.concatenate((p, \
    np.ones((batch_size,1))*config_wrf.constant_variable['ptop']),axis=1)
        
    play = (plev[:, 0:-1] + plev[:, 1:])/2                
    multi_feature[:, 5, :] = play
    
    multi_feature[:, 6, :] = th
    multi_feature[:, 7, :] = dz8w
    multi_feature[:, 8, :] = rho
    multi_feature[:, 9, :] = pi
    multi_feature[:, 10, :] = w0avg
    logger.info("yux 6")
    multi_feature[:, 11, :] = rthcuten
    multi_feature[:, 12, :] = rqvcuten
    multi_feature[:, 13, :] = rqccuten
    multi_feature[:, 14, :] = rqrcuten
    multi_feature[:, 15, :] = rqicuten
    multi_feature[:, 16, :] = rqscuten
    logger.info("yux 7")
          
    ### level pressure ###    
    auxiliary_variable = ['trigger', 'w0avg_output', 'p_diff', 'qv_sat', 'rh']
    auxiliary_feature = np.zeros([batch_size, len(auxiliary_variable), nlayers]) 
    logger.info("Shape of auxiliary_feature is {}".format(auxiliary_feature.shape))
    
    trigger = np.zeros(nca.shape)
    trigger[nca > 0.5] = 1    
    trigger = trigger.reshape((trigger.shape[0], 1))
    auxiliary_feature[:, 0, :] = trigger
    logger.info("Shape of trigger is {}".format(trigger.shape))
        
    W0AVGfctr = config_wrf.TST - 1
    W0fctr = 1 
    W0den = config_wrf.TST    
    w0avg_output = (w0avg * W0AVGfctr + w0 * W0fctr) / W0den        
    auxiliary_feature[:, 1, :] = w0avg_output
    logger.info("w0avg ok")              
    
    p_diff = plev[:, 0:-1] - plev[:, 1:]
    auxiliary_feature[:, 2, :] = p_diff
    logger.info("pdiff ok")
                                           
    logger.info("play shape {}, min {}, t shape {}, min{}".format(play.shape,play.min(), t.shape,t.min()))
                 
    qv_sat =  calculate_qv_saturated(play, t) 
    auxiliary_feature[:, 3, :] = qv_sat
    logger.info("qv_sat ok max {}".format(qv_sat.max()))
    
    rh = calculate_rh(qv, qv_sat)
    auxiliary_feature[:, 4, :] = rh        
    logger.info("rh ok max {}".format(rh.max()))

    logger.info("auxiliary done")
    
#==============================================================================
# apply data normalization
#==============================================================================
      
    logger.info("Apply data normalization")

    #apply data normalization to single height feature
    for variable_index, variable_name in enumerate(single_height_variable):

        if config_wrf.norm_method == 'z-score':
            single_feature[:, variable_index, :] = (single_feature[:, variable_index, 0] \
            - norm_mapping[variable_name]['mean']) / norm_mapping[variable_name]['scale']
            
        elif config_wrf.norm_method == 'min-max':
            single_feature[:, variable_index, 0] = single_feature[:, variable_index, 0] / \
            norm_mapping[variable_name]['max']
            
        elif config_wrf.norm_method == 'abs-max':                        
            single_feature[:, variable_index, 0] = single_feature[:, variable_index, 0] / \
            max( abs(norm_mapping[variable_name]['max']), abs(norm_mapping[variable_name]['min']))
                                  
    single_feature = single_feature.astype(np.float32)
    
    logger.info("end single level data normalization")
    #apply data normalization to multi height feature and multi height intergrated feature        
    for variable_index, variable_name in enumerate(multi_height_variable):        

#       logger.info("variable_name {}, max {}".format(variable_name, norm_mapping[variable_name]))
        
        if config_wrf.norm_method == 'z-score':                
            multi_feature[:, variable_index, :] = (multi_feature[:, variable_index, :] \
            - norm_mapping[variable_name]['mean']) / norm_mapping[variable_name]['scale']

        elif config_wrf.norm_method == 'min-max':
            multi_feature[:, variable_index, :] = multi_feature[:, variable_index, :] / \
            norm_mapping[variable_name]['max']
                         
        elif config_wrf.norm_method == 'abs-max':                        
            multi_feature[:, variable_index, :] = multi_feature[:, variable_index, :] / \
            max( abs(norm_mapping[variable_name]['max']), abs(norm_mapping[variable_name]['min']))
                                   
        # logger.info("end normalization variable_name {}".format(variable_name))
            
    multi_feature = multi_feature.astype(np.float32)
    logger.info("end multi level data normalization")
        
    for variable_index, variable_name in enumerate(auxiliary_variable):
#       logger.info("variable_name {}, max {}".format(variable_name, norm_mapping[variable_name]))
        
        if config_wrf.norm_method == 'z-score':                
            auxiliary_feature[:, variable_index, :] = (auxiliary_feature[:, variable_index, :] \
            - norm_mapping[variable_name]['mean']) / norm_mapping[variable_name]['scale']

        elif config_wrf.norm_method == 'min-max':
            auxiliary_feature[:, variable_index, :] = auxiliary_feature[:, variable_index, :] / \
            norm_mapping[variable_name]['max']
                         
        elif config_wrf.norm_method == 'abs-max':                        
            auxiliary_feature[:, variable_index, :] = auxiliary_feature[:, variable_index, :] / \
            max( abs(norm_mapping[variable_name]['max']), abs(norm_mapping[variable_name]['min']))
                                   
        # logger.info("end normalization variable_name {}".format(variable_name))
                
    auxiliary_feature = auxiliary_feature.astype(np.float32)
    logger.info("end auxiliary level data normalization")

    feature = np.concatenate( \
        [np.repeat(single_feature, nlayers, axis = 2), \
         multi_feature,\
         auxiliary_feature \
         ], 1)        

    logger.info("start to save feature.npy")
    # np.save("feature.npy", feature)            
          
    logger.info("Finish mskf_preprocess")
        
    feature_all_variable = single_height_variable + \
                           multi_height_variable + \
                           auxiliary_variable 
    
    return feature, feature_all_variable

