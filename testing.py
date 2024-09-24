
import unittest
from generated_code import is_palindrome

class TestIsPalindrome(unittest.TestCase):

    def test_palindrome_string(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal, Panama"))
        self.assertTrue(is_palindrome("Was it a car or a cat I saw"))
        self.assertTrue(is_palindrome("Able was I ere I saw Elba"))
        self.assertTrue(is_palindrome("A Santa at NASA"))

    def test_palindrome_string_with_spaces(self):
        self.assertFalse(is_palindrome("Hello World"))
        self.assertFalse(is_palindrome("   Hello World   "))

    def test_palindrome_string_with_punctuation(self):
        self.assertFalse(is_palindrome("Hello, World!"))
        self.assertFalse(is_palindrome("Hello World!"))

    def test_palindrome_string_with_numbers(self):
        self.assertFalse(is_palindrome("12321"))
        self.assertFalse(is_palindrome("123456"))

    def test_palindrome_string_with_empty_string(self):
        self.assertTrue(is_palindrome(""))

    def test_palindrome_string_with_non_string_input(self):
        with self.assertRaises(TypeError):
            is_palindrome(123)
        with self.assertRaises(TypeError):
            is_palindrome([1, 2, 3])
        with self.assertRaises(TypeError):
            is_palindrome({"a": 1, "b": 2})

    def test_palindrome_integer(self):
        self.assertTrue(is_palindrome(121))
        self.assertFalse(is_palindrome(123456))

if __name__ == '__main__':
    unittest.main()
