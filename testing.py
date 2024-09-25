
import unittest
from generated_code import reverse_string

class TestReverseString(unittest.TestCase):

    def test_reverse_string(self):
        self.assertEqual(reverse_string("hello"), "OLLEH")
        self.assertEqual(reverse_string("world"), "DLROW")
        self.assertEqual(reverse_string(""), "")
        self.assertEqual(reverse_string("a"), "A")
        self.assertEqual(reverse_string("abc"), "cba")
        self.assertEqual(reverse_string("123"), "321")
        self.assertEqual(reverse_string("123abc"), "cba321")
        self.assertEqual(reverse_string("123abc456"), "654321cba")
        self.assertEqual(reverse_string("123abc456def"), "fed654321cba")
        self.assertEqual(reverse_string("123abc456defghi"), "ihged654321cba")

    def test_non_string_input(self):
        with self.assertRaises(TypeError):
            reverse_string(123)
        with self.assertRaises(TypeError):
            reverse_string([1, 2, 3])
        with self.assertRaises(TypeError):
            reverse_string({"a": 1, "b": 2})

if __name__ == '__main__':
    unittest.main()
