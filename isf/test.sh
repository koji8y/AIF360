#!/bin/sh
set -x
python -m isf.tests.unit_test -v
#python -m isf.tests.unit_test -v TestStringMethods.test01_AdversarialDebiasing
#python -m isf.tests.unit_test -v TestStringMethods.test02_EqualizedOdds
#python -m isf.tests.unit_test -v TestStringMethods.test03_Massaging
#python -m isf.tests.unit_test -v TestStringMethods.test04_RejectOptionClassification
#python -m isf.tests.unit_test -v TestStringMethods.test05_Massaging_AA
