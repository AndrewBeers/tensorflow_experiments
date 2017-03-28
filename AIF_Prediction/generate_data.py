from qtim_tools.qtim_utilities.nifti_util import nifti_2_numpy
from qtim_tools.test_data import load

import nibabel as nib

def create_gradient_phantom(filepath, label_filepath):

    """ TO-DO: Fix the ktrans variation so that it correctly ends in 0.35, instead of whatever
        it currently ends in. Also parameterize and generalize everything for more interesting
        phantoms.
    """

    original_phantom_filepath = 

    original_phantom_data = nifti_2_numpy()

    print original_phantom_data

    AIF_subregion = original_phantom_data[:,70:,:].reshape(AIF_subregion.shape[0:-1] + (1,))
    AIF_subregion.mean(axis=0, dtype=np.float64)

    time_interval_seconds = float((11*60) / original_phantom_data.shape[-1])
    time_series = np.arange(0, original_phantom_data.shape[-1]) / (60 / time_interval_seconds)
    
    contrast_AIF = generate_contrast_agent_concentration(AIF,T1_tissue=1000, TR=5, flip_angle_degrees=30, injection_start_time_seconds=60, relaxivity=.0045, time_interval_seconds=time_interval_seconds, hematocrit=.45, T1_blood=1440)

    output_matrix_dims = (1000, 1000)
    ktrans_range = [.01, .5]
    ve_range = [.01, .5]

    groundtruth_data = np.zeros((output_matrix_dims + (2,)), dtype=float)
    generated_data = np.zeros((output_matrix_dims + (original_phantom_data.shape[-1],)), dtype=float)

    for ve_idx, ve in enumerate(np.arange(ve_range[0], (ve_range[1]-ve_range[0])/output_matrix_dims[1], ve_range[1])):
        for ktrans_idx, ktrans in enumerate(np.arange(.01, .35 +.35/60, .35/60)):
            gradient_nifti[ve_idx, ktrans_idx+10, 0] = float(ktrans)
            gradient_nifti[ve_idx, ktrans_idx+10, 1] = float(ve)
            time_nifti[ve_idx, ktrans_idx+10,:] = estimate_concentration([np.log(ktrans),-1 * np.log((1-ve)/ve)], contrast_AIF, time_series)
            print np.shape(time_nifti)
            print [ve_idx, ktrans_idx]
            print [ve, ktrans]

    nifti_util.save_numpy_2_nifti(time_nifti, filepath, 'gradient_toftsv6_concentration')
    time_nifti[:,10:70,:] = revert_concentration_to_intensity(data_numpy=time_nifti[:,10:70,:], reference_data_numpy=numpy_3d[:,10:70,:], T1_tissue=1000, TR=5, flip_angle_degrees=30, injection_start_time_seconds=60, relaxivity=.0045, time_interval_seconds=time_interval_seconds, hematocrit=.45, T1_blood=0, T1_map = [])

    nifti_util.save_numpy_2_nifti(gradient_nifti[:,:,0], filepath, 'gradient_toftsv6_ktrans_truth')
    nifti_util.save_numpy_2_nifti(gradient_nifti[:,:,1], filepath, 'gradient_toftsv6_ve_truth')
    nifti_util.save_numpy_2_nifti(time_nifti, filepath, 'gradient_toftsv6')