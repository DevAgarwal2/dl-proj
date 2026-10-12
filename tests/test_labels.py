import unittest

import numpy as np

from openmed.data.labels import (
    NIH_LABELS,
    labels_to_vector,
    normalize_label,
    parse_label_string,
)


class LabelTests(unittest.TestCase):
    def test_no_finding_is_empty(self):
        self.assertEqual(parse_label_string("No Finding"), [])
        self.assertEqual(parse_label_string(""), [])
        self.assertEqual(parse_label_string(None), [])

    def test_multi_label_parsing(self):
        labels = parse_label_string("Cardiomegaly|Effusion|Pneumonia")
        self.assertEqual(labels, ["Cardiomegaly", "Effusion", "Pneumonia"])

    def test_alias_normalisation(self):
        self.assertEqual(parse_label_string("Pleural Effusion"), ["Effusion"])
        self.assertEqual(normalize_label("enlarged heart"), "Cardiomegaly")
        self.assertEqual(normalize_label("Pneumonia"), "Pneumonia")

    def test_vector_encoding(self):
        vector = labels_to_vector(["Cardiomegaly", "Pneumonia"])
        self.assertEqual(vector.shape, (len(NIH_LABELS),))
        self.assertEqual(vector.sum(), 2.0)
        self.assertEqual(vector[NIH_LABELS.index("Cardiomegaly")], 1.0)
        self.assertEqual(vector[NIH_LABELS.index("Pneumonia")], 1.0)
        self.assertEqual(vector.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
