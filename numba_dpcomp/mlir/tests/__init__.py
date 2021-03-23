from numba.testing import unittest
from numba.testing import load_testsuite
from os.path import dirname

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    this_dir = dirname(__file__)
    suite.addTests(load_testsuite(loader, this_dir))

    return suite
