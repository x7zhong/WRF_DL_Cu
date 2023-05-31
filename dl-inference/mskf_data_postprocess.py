#postprocess rrtmg DL model outputs to get heating rate

import numpy as np
import logging

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

def unnormalized(predicts, norm_mapping, label_all_variable, norm_method):
    
    logger = generate_logger("./mskf_postprocess.log")

    logger.info("Start unnormalized")
    
    predicts_unnorm = np.zeros(predicts.shape)
    # logger.info("label_all_variable {}".format(label_all_variable))
    # logger.info("norm_mapping {}".format(norm_mapping))
    
    for index, variable_name in enumerate(label_all_variable):
        
        # logger.info("start denormalize variable_name {}".format(variable_name))
        # logger.info("max {}".format(norm_mapping[variable_name]))
        
        if norm_method == 'z-score':                                
            
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            norm_mapping[variable_name]["scale"] + \
            norm_mapping[variable_name]["mean"]
            
        elif norm_method == 'min-max':                                
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            norm_mapping[variable_name]["max"]

        elif norm_method == 'abs-max':   
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            max( abs(norm_mapping[variable_name]['max']), \
            abs(norm_mapping[variable_name]['min']) )
            
        # logger.info("end denormalization variable_name {}".format(variable_name))
                
    return predicts_unnorm 