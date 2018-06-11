"""UnityEyes data source for gaze estimation."""
import os
from threading import Lock

import cv2 as cv
import numpy as np
import tensorflow as tf
import ujson

from core import BaseDataSource
import util.gaze


class UnityEyes(BaseDataSource):
    """UnityEyes data loading class."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 unityeyes_path: str,
                 testing=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'UnityEyes'
        if testing:
            self._short_name += ':test'

        # Create global index over all specified keys
        self._images_path = unityeyes_path
        self._file_stems = sorted([p[:-5] for p in os.listdir(unityeyes_path)
                                   if p.endswith('.json')])
        self._num_entries = len(self._file_stems)

        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size, testing=testing, **kwargs)

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source UnityEyes."""
        return self._short_name

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from UnityEyes."""
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

                file_stem = self._file_stems[current_index]
                jpg_path = '%s/%s.jpg' % (self._images_path, file_stem)
                json_path = '%s/%s.json' % (self._images_path, file_stem)
                with open(json_path, 'r') as f:
                    json_data = ujson.load(f)
                entry = {
                    'full_image': cv.imread(jpg_path, cv.IMREAD_GRAYSCALE),
                    'json_data': json_data,
                }
                assert entry['full_image'] is not None
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Use annotations to segment eyes and calculate gaze direction."""
        full_image = entry['full_image']
        json_data = entry['json_data']

        ih, iw = full_image.shape
        oh, ow = 18, 30

        def process_coords(coords_list):
            coords = [eval(l) for l in coords_list]
            return np.array([(x, ih-y, z) for (x, y, z) in coords])
        interior_landmarks = process_coords(json_data['interior_margin_2d'])
        caruncle_landmarks = process_coords(json_data['caruncle_2d'])

        # Prepare to segment eye image
        left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
        right_corner = interior_landmarks[8, :2]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                              np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

        # Centre axes to eyeball centre
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-iw/2], [-ih/2]]

        # Scale image to fit output dimensions
        scale_mat = np.asmatrix(np.eye(3))
        np.fill_diagonal(scale_mat, ow / eye_width)

        # Re-centre eye image such that eye fits (based on determined `eye_middle`)
        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = iw/2 - eye_middle[0] + 0.5 * eye_width
        recentre_mat[1, 2] = ih/2 - eye_middle[1] + 0.5 * oh / ow * eye_width

        # Apply transforms
        transform_mat = recentre_mat * scale_mat * translate_mat
        eye = cv.warpAffine(full_image, transform_mat[:2, :3], (ow, oh))

        # Convert look vector to gaze direction in polar angles
        look_vec_ = np.array(eval(json_data['eye_details']['look_vec']))[:3]
        look_vec_[0] = -look_vec_[0]
        gaze = util.gaze.vector_to_pitchyaw(look_vec_.reshape((1, 3))).flatten()
        if gaze[1] > 0.0:
            gaze[1] = np.pi - gaze[1]
        elif gaze[1] < 0.0:
            gaze[1] = -(np.pi + gaze[1])

        # Preprocessing for NN
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, axis=0)  # Images are expected to have 3 dimensions
        entry['eye'] = eye

        return {
            'eye': eye.astype(np.float32),
            'gaze': gaze.astype(np.float32),
        }
