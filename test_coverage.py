import coverage
import unittest

cov = coverage.Coverage(source=["generated_code"])
cov.start()

loader = unittest.TestLoader()
suite = loader.discover('.', pattern='testing.py')

runner = unittest.TextTestRunner()
result = runner.run(suite)

cov.stop()
cov.save()

print("\nCoverage Report:")
cov.report()

if not result.wasSuccessful():
    exit(1)


# Coverage Report:
# Name                Stmts   Miss  Cover
# ---------------------------------------
# generated_code.py       8      1    88%
# ---------------------------------------
# TOTAL                   8      1    88%
#
# Process finished with exit code 1