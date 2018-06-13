"""HDF5 data source for gaze estimation."""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf
import os
from core.data_source import  BaseDataSource

import logging
import pandas as pd


logger = logging.getLogger(__name__)


class BaseHDF5Source(BaseDataSource):
    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        self._hdf5.close()
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from HDF5."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                key, index = self._index_to_key[current_index]
                data = self._hdf5[key]
                entry = {}
                for name in ('eye', 'gaze', 'head'):
                    if name in data:
                        entry[name] = data[name][index, :]
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Resize eye image and normalize intensities."""
        eye = entry['eye']
        eye = cv.resize(eye, (30, 18))
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        # eye = np.expand_dims(eye, axis=0)  # Images are expected to have 3 dimensions  # use this line if using GPU
        eye = np.expand_dims(eye, axis=-1)  # Images are expected to have 3 dimensions
        entry['eye'] = eye

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry


class HDF5Source(BaseHDF5Source):
    """HDF5 data loading class (using h5py)."""
    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 keys_to_use: List[str],
                 hdf_path: str,
                 testing=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            self._short_name += ':test'

        # Create global index over all specified keys
        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = hdf5[key]['eye'].shape[0]
            logger.info("number of eyes: {}".format(n))
            for i in range(n):
                self._index_to_key[index_counter] = (key, i)
                index_counter += 1
        self._num_entries = index_counter

        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0


class HDF5SourceRaw(HDF5Source):

    def preprocess_entry(self, entry):
        # entries are in float64, but tensor expects float32
        entry['eye'] = entry['eye'].astype(np.float32)
        entry['gaze'] = entry['gaze'].astype(np.float32)
        return entry


class BootstrappedHDF5Source(BaseHDF5Source):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 keys_to_use: List[str],
                 hdf_path: str,
                 random_seed: int,
                 model_identifier: str,
                 testing=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys.
        This is nearly the same __init__() as in HDF5Source. The only change is how i in range(n) is sampled."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            self._short_name += ':test'

        # set the random seed for reproduceability
        np.random.seed(random_seed)

        # Create global index over all specified keys
        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = hdf5[key]['eye'].shape[0]
            logger.info("number of eyes: {}".format(n))
            bootstrapped_indices = list(np.random.randint(0, n, n))
            logger.info("Bootstrapped {} indices".format(n))
            for i in bootstrapped_indices:
                self._index_to_key[index_counter] = (key, i)
                index_counter += 1
        # Keep track of training indices (use for out-of-bag-error calculation)
        self.persist_indices(self._index_to_key, model_identifier)

        self._num_entries = index_counter

        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    def persist_indices(self, index_to_key_dict: dict, model_identifier):
        """
        Writes the index_to_key_dict to a csv file. Creates a folder in .../outputs/ called model_identifier. In that
        folder a csv file called "train_indices.csv" will contain the index_to_key_dict.
        :param index_to_key_dict: self._index_to_key
        :param model_identifier: e.g. DenseBag_RS009_123451234
        :return: None
        """
        logger.info("Storing training indices")

        # dicts are not sorted. We now sort it to make sure to keep the correct pairs.
        keys_sorted = sorted(index_to_key_dict.keys())
        # keys is either "train" or "validation"
        keys = [index_to_key_dict[i][0] for i in range(len(keys_sorted))]
        # index of the example in the dataset
        index = [index_to_key_dict[i][1] for i in range(len(keys_sorted))]
        df = pd.DataFrame(data={'key': keys, 'index': index}, index=keys_sorted)

        # We call this script from the folder .../src/
        outputs_path = '../outputs/{}/'.format(model_identifier)
        # The folder doesn't exist yet because we have not started training.
        os.mkdir(outputs_path)
        path_out = os.path.join(outputs_path, 'train_indices.csv')
        df.to_csv(path_out)


class ValidationSetHDF5Source(BaseHDF5Source):
    """ Loads the validation data
    @pre: there is a csv file with training indices in output folder for this model (train_indices.csv)
    """

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 keys_to_use: List[str],
                 hdf_path: str,
                 model_identifier: str,
                 testing=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys.
        This is nearly the same __init__() as in HDF5Source. The only change is how i in range(n) is sampled."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            self._short_name += ':test'

        training_indices = self.load_training_indices(model_identifier)
        logger.info("Number of training samples: {}".format(len(training_indices)))

        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = hdf5[key]['eye'].shape[0]
            for i in range(n):
                tuple_identifier = (key, i)
                if tuple_identifier not in training_indices:
                    self._index_to_key[index_counter] = tuple_identifier
                    index_counter += 1

                # debug
                if len(self._index_to_key) > 10:
                    break


        self._num_entries = index_counter
        logger.info("Validation Set has {} entries (training set has {} entries)".format(self._num_entries, len(training_indices)))

        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    def load_training_indices(self, model_identifier):
        """
        Returns list with tuples
        :param model_identifier:
        :return: [('train', 2732), ('train', 10799), ('train', 9845),...]
        """
        # We call this script from the folder .../src/
        path = '../outputs/{}/train_indices.csv'.format(model_identifier)
        df_indices = pd.read_csv(path)

        key_indey_pairs = df_indices.apply(lambda row: (row['key'], row['index']), axis=1)
        # 0             (train, 2732)
        # 1            (train, 10799)
        # 2             (train, 9845)
        return key_indey_pairs.values.tolist()


