import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from fastbev_service.tensor import TensorFormatError, TensorSpec, load_tensor, validate_tensor


DTYPE_CODES = {
    "float32": 3,
    "float16": 2,
    "int32": 1,
    "int64": 4,
    "uint64": 5,
    "uint32": 6,
    "int8": 7,
    "uint8": 8,
}


def encode_tensor(array: np.ndarray) -> bytes:
    dtype_code = DTYPE_CODES[str(array.dtype)]
    header = np.array([0x33FF1101, array.ndim, dtype_code], dtype=np.int32).tobytes()
    dims = np.array(array.shape, dtype=np.int32).tobytes()
    return header + dims + array.tobytes()


class TensorParserTest(unittest.TestCase):
    def test_loads_float32_tensor(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        loaded = load_tensor(encode_tensor(source))
        np.testing.assert_array_equal(loaded, source)
        self.assertEqual(loaded.dtype, np.dtype("<f4"))
        self.assertEqual(loaded.shape, (3, 4))

    def test_loads_int64_tensor(self):
        source = np.arange(8, dtype=np.int64).reshape(2, 4)
        loaded = load_tensor(encode_tensor(source))
        np.testing.assert_array_equal(loaded, source)
        self.assertEqual(loaded.dtype, np.dtype("<i8"))

    def test_rejects_bad_magic(self):
        source = bytearray(encode_tensor(np.arange(4, dtype=np.float32)))
        source[0:4] = np.array([0], dtype=np.int32).tobytes()
        with self.assertRaises(TensorFormatError):
            load_tensor(source)

    def test_rejects_payload_size_mismatch(self):
        source = encode_tensor(np.arange(4, dtype=np.float32))[:-1]
        with self.assertRaises(TensorFormatError):
            load_tensor(source)

    def test_validate_tensor_rejects_wrong_dtype(self):
        array = np.zeros((6, 160000), dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_tensor(array, TensorSpec("valid_c_idx", np.dtype("<f4"), (6, 160000)))

    def test_validate_tensor_rejects_wrong_shape(self):
        array = np.zeros((1, 160000), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_tensor(array, TensorSpec("valid_c_idx", np.dtype("<f4"), (6, 160000)))


if __name__ == "__main__":
    unittest.main()
