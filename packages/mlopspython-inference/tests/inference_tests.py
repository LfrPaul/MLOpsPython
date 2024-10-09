import unittest
from pathlib import Path
import logging
from unittest.mock import MagicMock

import numpy as np

from mlopspython_inference.inference_pillow import Inference, IModel

BASE_PATH = Path(__file__).resolve().parent
output_directory = BASE_PATH / "output"
input_directory = BASE_PATH / "input"

class TestInference(unittest.TestCase):

    def test_inference(self):
        mock = MagicMock(IModel)
        mock.predict.return_value = np.array([[1.0, 2.370240289845506e-30, 0.0]])

        inference = Inference(logging, mock)
        inference_result = inference.execute(str(input_directory / "images" / "cat.png"))

        expected_result = {'prediction': 'Cat', 'values': [1.0, 2.370240289845506e-30, 0.0]}
        self.assertEqual(inference_result['prediction'], expected_result['prediction'])
        self.assertEqual(len(inference_result['values']), len(expected_result['values']))


if __name__ == "__main__":
    unittest.main()
