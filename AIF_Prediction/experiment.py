from model import Model

if __name__ == '__main__':

    # Recurrent
    # test_model = Model(max_seq_len=65,
    # patch_x=5,
    # patch_y=5,
    # num_classes=2,
    # cnn_features_out = 32,
    # num_hidden = 32,
    # num_layers = 4,
    # model_type = 'lstm',
    # optimizer_type="regression",
    # n_samples_train_test=[5000,10000],
    # total_epochs=300000,
    # batch_size=400,
    # display_epoch=10,
    # test_batch_size=10000,
    # load_data=True,
    # old_model=False,
    # train=True,
    # test=False,
    # reconstruct=False,
    # dce_filepath=None,
    # ktrans_filepath=None,
    # ve_filepath=None,
    # output_test_results='results.csv',
    # output_model='model',
    # output_ktrans_filepath='ktrans.nii.gz',
    # output_ve_filepath='ve.nii.gz')

    # cnnRNN
    test_model = Model(max_seq_len=65,
    patch_x=5,
    patch_y=5,
    num_classes=2,
    cnn_filters=30,
    cnn_features_out = 32,
    num_hidden_lstm = 128,
    num_layers = 4,
    model_type = 'lstm',
    optimizer_type="regression",
    n_samples_train_test=[5000,3000],
    total_epochs=300000,
    batch_size=400,
    display_epoch=10,
    test_batch_size=10000,
    load_data=True,
    old_model=False,
    train=True,
    test=True,
    reconstruct=False,
    dce_filepath=None,
    ktrans_filepath=None,
    ve_filepath=None,
    output_test_results='results.csv',
    output_model='model',
    output_ktrans_filepath='ktrans.nii.gz',
    output_ve_filepath='ve.nii.gz')


    test_model.run_model()