import unittest
from pytorch_model_parser import parse_model
import torchvision.models as models


class TestParser(unittest.TestCase):
    def test_resnet18(self):
        model = models.resnet18()
        details = parse_model(model)
        self.assertTrue(len(details) > 0)
        # Add more specific tests based on the expected output


if __name__ == "__main__":
    unittest.main()
