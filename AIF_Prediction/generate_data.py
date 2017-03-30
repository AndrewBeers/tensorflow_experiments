from __future__ import division
from qtim_tools.qtim_utilities.nifti_util import nifti_2_numpy
from qtim_tools.test_data.load import load_test_file

import math
import nibabel as nib
import numpy as np

def create_gradient_phantom():

    """ TO-DO: Fix the ktrans variation so that it correctly ends in 0.35, instead of whatever
        it currently ends in. Also parameterize and generalize everything for more interesting
        phantoms.
    """

    original_phantom_filepath = load_test_file('dce_tofts_v6')

    original_phantom_data = nifti_2_numpy(original_phantom_filepath)

    AIF_subregion = original_phantom_data[:,70:,:]
    AIF_subregion = AIF_subregion.reshape(np.product(AIF_subregion.shape[0:-1]), AIF_subregion.shape[-1])
    AIF = AIF_subregion.mean(axis=0, dtype=np.float64)

    time_interval_seconds = float((11*60) / original_phantom_data.shape[-1])
    time_series = np.arange(0, original_phantom_data.shape[-1]) / (60 / time_interval_seconds)
    
    contrast_AIF = convert_intensity_to_concentration(AIF,T1_tissue=1000, TR=5, flip_angle_degrees=30, injection_start_time_seconds=60, relaxivity=.0045, time_interval_seconds=time_interval_seconds, hematocrit=.45, T1_blood=1440)

    output_matrix_dims = (300, 300)
    ktrans_range = [.01, .5]
    ve_range = [.01, .5]

    groundtruth_data = np.zeros((output_matrix_dims + (2,)), dtype=float)
    generated_data = np.zeros((output_matrix_dims + (original_phantom_data.shape[-1],)), dtype=float)

    print generated_data.shape
    # print np.arange(ve_range[0], (ve_range[1]-ve_range[0])/output_matrix_dims[1], ve_range[1])

    for ve_idx, ve in enumerate(np.arange(ve_range[0], ve_range[1], (ve_range[1]-ve_range[0])/output_matrix_dims[1])):
        for ktrans_idx, ktrans in enumerate(np.arange(ktrans_range[0], ktrans_range[1],  (ktrans_range[1]-ktrans_range[0])/output_matrix_dims[1])):
            
            groundtruth_data[ktrans_idx, ve_idx, :] = [ktrans, ve]
            generated_data[ktrans_idx, ve_idx, :] = np.array(estimate_concentration([np.log(ktrans),-1 * np.log((1-ve)/ve)], contrast_AIF, time_interval_seconds))
            print [ve_idx, ktrans_idx]
            print [ve, ktrans]

    generated_data = revert_concentration_to_intensity(data_numpy=generated_data, reference_data_numpy=original_phantom_data[:,10:70,:], T1_tissue=1000, TR=5, flip_angle_degrees=30, injection_start_time_seconds=60, relaxivity=.0045, time_interval_seconds=time_interval_seconds, hematocrit=.45, T1_blood=0, T1_map = [])

    np.save('Generated_AIF_Intensities.npy', generated_data)
    np.save('Groundtruth_AIF_Intensities.npy', groundtruth_data)


def estimate_concentration(params, contrast_AIF_numpy, time_interval):

    # Notation is very inexact here. Clean it up later.

    estimated_concentration = [0]
    # if params[0] > 10 or params[1] > 10:
    #   return estimated_concentration

    append = estimated_concentration.append
    e = math.e

    ktrans = params[0]
    ve = params[1]
    kep = ktrans / ve

    log_e = -1 * kep * time_interval
    capital_E = e**log_e
    log_e_2 = log_e**2

    block_A = (capital_E - log_e - 1)
    block_B = (capital_E - (capital_E * log_e) - 1)
    block_ktrans = ktrans * time_interval / log_e_2

    for i in xrange(1, np.size(contrast_AIF_numpy)):
        term_A = contrast_AIF_numpy[i] * block_A
        term_B = contrast_AIF_numpy[i-1] * block_B
        append(estimated_concentration[-1]*capital_E + block_ktrans * (term_A - term_B))

    # Quick, error prone convolution method
    # print estimated_concentration
        # res = np.exp(-1*kep*time_series)
        # estimated_concentration = ktrans * np.convolve(contrast_AIF_numpy, res) * time_series[1]
        # estimated_concentration = estimated_concentration[0:np.size(res)]

    return estimated_concentration

def revert_concentration_to_intensity(data_numpy, reference_data_numpy, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=0, T1_map = []):

    if T1_map != []:
        R1_pre = 1 / T1_map
        R1_pre = np.reshape(R1_pre.shape + (1,))
    else:
        R1_pre = 1 / T1_tissue

    flip_angle_radians = flip_angle_degrees*np.pi/180
    a = np.exp(-1 * TR * R1_pre)
    relative_term = (1-a) / (1-a*np.cos(flip_angle_radians))

    if len(reference_data_numpy.shape) == 1:
        baseline = np.mean(reference_data_numpy[0:int(np.round(injection_start_time_seconds/time_interval_seconds))])
        baseline = np.tile(baseline, reference_data_numpy.shape[-1])
    if len(reference_data_numpy.shape) == 2:
        baseline = np.mean(reference_data_numpy[:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=1)
        baseline = np.tile(np.reshape(baseline, (baseline.shape[0], 1)), (1,reference_data_numpy.shape[-1]))
    if len(reference_data_numpy.shape) == 3:

        baseline = np.mean(np.mean(reference_data_numpy[:,:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=2))
    if len(reference_data_numpy.shape) == 4:
        baseline = np.mean(reference_data_numpy[:,:,:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=3)
        baseline = np.tile(np.reshape(baseline, (baseline.shape[0],baseline.shape[1],baseline.shape[2], 1)), (1,1,1,reference_data_numpy.shape[-1]))

    data_numpy = np.exp(data_numpy / (-1 / (relaxivity*TR)))
    data_numpy = (data_numpy * a -1) / (data_numpy * a * np.cos(flip_angle_radians) - 1)
    data_numpy = data_numpy / relative_term
    data_numpy = data_numpy * baseline
    ###

    return data_numpy

def convert_intensity_to_concentration(data_numpy, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=0, T1_map = []):

    old_settings = np.seterr(divide='ignore', invalid='ignore')

    flip_angle_radians = flip_angle_degrees*np.pi/180

    if T1_map != []:
        R1_pre = float(1) / float(T1_map)
        R1_pre = np.reshape(R1_pre.shape + (1,))
    elif T1_blood == 0:
        R1_pre = float(1) / float(T1_tissue)
    else:
        R1_pre = float(1) / float(T1_blood)

    a = np.exp(-1 * TR * R1_pre)
    relative_term = (1-a) / (1-a*np.cos(flip_angle_radians))

    dim = len(data_numpy.shape)

    if dim == 1:
        baseline = np.mean(data_numpy[0:int(np.round(injection_start_time_seconds/time_interval_seconds))])
        baseline = np.tile(baseline, data_numpy.shape[-1])
    elif dim > 1 and dim < 5:
        baseline = np.mean(data_numpy[...,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=dim-1)
        baseline = np.tile(np.reshape(baseline, (baseline.shape[0:dim-1] + (1,))), (1,)*(dim-1) + (data_numpy.shape[-1],))
    else:
        print 'Dimension error. Please enter an array with dimensions between 1 and 4.'

    output_numpy = np.copy(data_numpy)

    output_numpy = np.nan_to_num(output_numpy / baseline)

    output_numpy = output_numpy * relative_term

    output_numpy = (output_numpy - 1) / (a * (output_numpy * np.cos(flip_angle_radians) - 1))

    output_numpy[output_numpy < 0] = 0

    output_numpy = -1 * (1 / (relaxivity * TR)) * np.log(output_numpy)

    output_numpy = np.nan_to_num(output_numpy)

    np.seterr(**old_settings)

    if T1_blood == 0:
        return output_numpy
    else:
        output_numpy = output_numpy / (1-hematocrit)
        return output_numpy

if __name__ == '__main__':
    create_gradient_phantom()