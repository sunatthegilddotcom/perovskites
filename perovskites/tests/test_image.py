import os
import numpy as np
import unittest

from perovskites.utils import image_loader as loader
from perovskites.utils import image_processing as process
from perovskites.utils import miscellaneous as misc

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
data_path = os.path.join(parent_dir, 'data')
sample_path = os.path.join(data_path, 'sample_data.pickle')
tif_path = os.path.join(data_path, "sample_stack.tif")
IMG = process.read_image(tif_path)
DATA = loader._get_data_from_pickle(sample_path)


class test_image_loader(unittest.TestCase):

    def test_sample(self):
        """
        Tests if the PLDataLoader() class correctly samples PL image data
        """
        self.assertIsInstance(DATA, tuple)
        self.assertTrue(DATA[0].shape == (9, 32, 32, 1))

    def test_create_pickle(self):
        """
        Tests if data can be correctly transfered into a pickle file.
        """
        pickle_path = os.path.join(data_path, 'test_pickle.pickle')
        loader._create_data_pickle(DATA, pickle_path)
        data2 = loader._get_data_from_pickle(pickle_path)
        for i in range(len(DATA)):
            self.assertTrue(DATA[i].any() == data2[i].any())


class test_image_processer(unittest.TestCase):

    def test_read_image(self):
        """
        Tests if read_image() function properly transforms tif file into
        skimage processable numpy array.
        """
        self.assertEqual(len(IMG.shape), 3)
        self.assertEqual(IMG.shape, (512, 512, 50))

    def test_mean_over_depth(self):
        """
        Tests if mean_over_depth() accurately calculates array mean based
        on the 3rd axis (time-depth)
        """
        test1 = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
        test2 = np.array([[1, 2], [3, 4]])
        result1 = process.mean_over_depth(test1)
        result2 = process.mean_over_depth(test2)
        expected = np.array([[1., 2.], [3., 4.]])
        size = len(process.mean_over_depth(IMG)[0])
        self.assertTrue(result1.any(), expected.any())
        self.assertTrue(result2.any(), test2.any())
        self.assertEqual(size, 512)

    def test_normalize(self):
        """
        Tests if normalize() function correctly adjusts pixel values
        between 0 & 1 (white & black) for a given numpy array.
        """
        test = np.array([[0, 1], [3, 4]])
        result = process.normalize(test)
        expected = np.array([[[0.0, 0.25], [.75, 1.]]])
        self.assertTrue(result.any(), expected.any())


class test_miscellaneous(unittest.TestCase):

    def test_booleanize(self):
        """
        Determines if booleanize() correctly converts boolean strings into
        boolean types.
        """
        test = {'a': 'True', 'b': 'False', 'c': 'true', 'd': 'false'}
        result = misc.booleanize(test)
        self.assertIsInstance(result, dict)
        for key in result.keys():
            self.assertIsInstance(result[key], bool)
