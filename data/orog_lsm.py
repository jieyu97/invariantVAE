import numpy as np
import torch


def apply_normalization(data):
    mins = data.ravel().min()
    maxs = data.ravel().max()
    x = data
    return (x - mins) / (maxs - mins)
    
def orog_lsm_data(model_type = 'ivae3d'):

    # model_type: ivae3d or ae
    orog = np.load('/hkfs/home/haicore/econ/gm2154/ae_project/data/orog.npy')
    lsm = np.load('/hkfs/home/haicore/econ/gm2154/ae_project/data/lsm.npy')
    orog_normalized = apply_normalization(orog)
    
    if model_type == 'ivae3d':
    
        orog_reshape = np.expand_dims(orog_normalized, axis=(0,1,2))
        lsm_reshape = np.expand_dims(lsm, axis=(0,1,2))
        # (1,1,1,81,81)
        
        orog_input = torch.from_numpy(orog_reshape)
        lsm_input = torch.from_numpy(lsm_reshape)
        
        orog_lsm = torch.cat([orog_input, lsm_input], dim=1)
        # (1,2,1,81,81)
        
    elif model_type == 'ae':
    
        orog_reshape = np.expand_dims(orog_normalized, axis=(0,1))
        lsm_reshape = np.expand_dims(lsm, axis=(0,1))
        # (1,1,81,81)
        
        orog_input = torch.from_numpy(orog_reshape)
        lsm_input = torch.from_numpy(lsm_reshape)
        
        orog_lsm = torch.cat([orog_input, lsm_input], dim=1)
        # (1,2,81,81)

    return orog_lsm