import unittest

from perovskites.utils import image_loader as loader


class test_image_loader(unittest.TestCase):

    def __init__(self):
        """
        Initializes class
        """
        self.dataset = loader.PLDataLoader()

    def test_predict(self):
        """
        Tests if the PLDataLoader() class correctly samples PL image data
        """
        sample_path = "drive/Shareddrives/Perovskites_DIRECT/" + (
          "sample_image_Ld80_data.pickle")
        dataset = self.dataset
        sample_data = dataset.sample(frac=0.008)
        sample_data2 = loader._get_data_from_pickle(sample_path)
        self.assertIsInstance(sample_data, tuple)
        self.assertTrue(sample_data[0].shape[0] == 9)
        for i in range(len(sample_data)):
            self.assertTrue(sample_data[i].any() == sample_data2[i].any())

    def test_train_test_split(self):
        """
        Tests if the data is split properly in the train_test_split() function.
        """
        split = self.dataset.train_test_split()
        X_train, X_test, y_train, y_test = split
        self.assertTrue(len(split) == 4)
        self.assertTrue(len(X_train) + len(X_test) == 1245)
        self.assertTrue(len(y_train) + len(y_test) == 1245)
        self.assertTrue(len(X_test) == 249)
