

def get_model(Model, session, learning_rate, batch_size, hdf_path, Hdf5Source):
    """

    :param Model:
    :param session:
    :param learning_rate: 1e-3
    :param batch_size: 128
    :param num_epochs: 100
    :param hdf_path: '../datasets/MPIIGaze_kaggle_students.h5'
    :return:
    """
    model = Model(

        # Tensorflow session
        # Note: The same session must be used for the model and the data sources.
        session,

        # The learning schedule describes in which order which part of the network should be
        # trained and with which learning rate.
        #
        # A standard network would have one entry (dict) in this argument where all model
        # parameters are optimized. To do this, you must specify which variables must be
        # optimized and this is done by specifying which prefixes to look for.
        # The prefixes are defined by using `tf.variable_scope`.
        #
        # The loss terms which can be specified depends on model specifications, specifically
        # the `loss_terms` output of `BaseModel::build_model`.
        learning_schedule=[
            {
                'loss_terms_to_optimize': {
                    'gaze_mse': ['conv', 'fc'],
                },
                'metrics': ['gaze_angular'],
                'learning_rate': learning_rate,
            },
        ],

        test_losses_or_metrics=['gaze_mse', 'gaze_angular'],

        # Data sources for training and testing.
        train_data={
            'real': Hdf5Source(
                session,
                batch_size,
                hdf_path=hdf_path,
                keys_to_use=['train'],
                min_after_dequeue=100,
            ),
        },
        test_data={
            'real': Hdf5Source(
                session,
                batch_size,
                hdf_path=hdf_path,
                keys_to_use=['validation'],
                testing=True,
            ),
        },
    )
    return model

