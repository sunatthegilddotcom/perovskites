import os
import sys
import unittest

import pandas as pd
from .utils import codebase

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
data_path = os.path.join(parent_dir, 'data')


class test_Perceptron(unittest.TestCase):

    def test_predict(self):
        """
        Testing the perceptron
        """
        weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
        df = pd.read_csv(os.path.join(data_path, 'test.csv'))
        dataset = df.values.tolist()

        for row in dataset:
            prediction = codebase.predict(row, weights)
            self.assertAlmostEqual(row[-1], prediction)
