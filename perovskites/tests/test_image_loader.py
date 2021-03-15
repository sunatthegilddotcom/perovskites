import os
import unittest

from perovskites.utils import image_loader as loader

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
data_path = os.path.join(parent_dir, 'data')
sample_path = os.path.join(data_path, 'sample_data.pickle')


class test_image_loader(unittest.TestCase):

    def __init__(self):
        """
        Establishes sample data.
        """
        self.data = loader._get_data_from_pickle(sample_path)

    def test_sample(self):
        """
        Tests if the PLDataLoader() class correctly samples PL image data
        """
        self.assertIsInstance(self.data, tuple)
        self.assertTrue(self.data[0].shape == (9, 32, 32, 1))

    # def test_train_test_split(self):
    #     """
    #     Tests if the data is split properly in the train_test_split() function.
    #     """
    #     split = self.data.train_test_split()
    #     self.assertTrue(len(split) == 4)
    #     X_train, X_test, y_train, y_test = split
    #     self.assertTrue(len(X_train) + len(X_test) == 9)
    #     self.assertTrue(len(y_train) + len(y_test) == 9)
    #     self.assertTrue(len(X_test) == 1)
