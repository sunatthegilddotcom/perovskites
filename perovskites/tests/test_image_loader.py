import os
import unittest

from perovskites.utils import image_loader as loader
from perovskites.utils import image_processing as process
from perovskites.utils import miscellaneous as misc

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
data_path = os.path.join(parent_dir, 'data')
sample_path = os.path.join(data_path, 'sample_data.pickle')


class test_image_loader(unittest.TestCase):

    def test_sample(self):
        """
        Tests if the PLDataLoader() class correctly samples PL image data
        """
        self.data = loader._get_data_from_pickle(sample_path)
        self.assertIsInstance(self.data, tuple)
        self.assertTrue(self.data[0].shape == (9, 32, 32, 1))

    def test_create_pickle(self):
        """
        Tests if data can be correctly transfered into a pickle file.
        """
        pickle_path = os.path.join(data_path, 'test_pickle.pickle')
        loader._create_data_pickle(self.data, pickle_path)
        data2 = loader._get_data_from_pickle(pickle_path)
        for i in range(len(self.data)):
          self.assertTrue(self.data[i].any() == data2[i].any())

class test_image_processer(unittest.TestCase):
    def test_read_image(self):
        """
        Tests read_image() function.
        """

    def test_mean_over_depth(self):
        """
        Tests mean_over_depth() function.
        """

    def test_normalize(self):
        """
        Tests normalize() function.
        """

    def test_crop_image(self):
        """
        Tests crop_image() function.
        """

    def test_get_channel(self):
        """
        Tests get_channel() function.
        """

    def test_resize_base_on_FOV(self):
        """
        Tests resize_based_on_FOV() function.
        """

    def test_img_as_feed(self):
        """
        Tests img_as_feed() function.
        """


class test_miscellaneous(unittest.TestCase):

    def test_booleanize(self):
        """
        Tests booleanize() function.
        """

    def test_convert_to_long_path(self):
        """
        Tests convert_to_long_path() function.
        """

    def test_best_rowcol_split(self):
        """
        Tests best_rowcol_split() function.
        """
