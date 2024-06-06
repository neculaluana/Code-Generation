
import unittest
from generated_code import fibonacci

class TestFibonacci(unittest.TestCase):

    def test_fibonacci_zero(self):
        self.assertEqual(fibonacci(0), [])

    def test_fibonacci_one(self):
        self.assertEqual(fibonacci(1), [0])

    def test_fibonacci_two(self):
        self.assertEqual(fibonacci(2), [0, 1])

    def test_fibonacci_three(self):
        self.assertEqual(fibonacci(3), [0, 1, 1])

    def test_fibonacci_four(self):
        self.assertEqual(fibonacci(4), [0, 1, 1, 2])

    def test_fibonacci_five(self):
        self.assertEqual(fibonacci(5), [0, 1, 1, 2, 3])

    def test_fibonacci_negative(self):
        with self.assertRaises(ValueError):
            fibonacci(-1)

if __name__ == '__main__':
    unittest.main()