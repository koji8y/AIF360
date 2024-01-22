#!/bin/bash
test_class="TestStringMethods"
test_func=("test01_AdversarialDebiasing" "test02_EqualizedOdds" "test03_Massaging" "test04_RejectOptionClassification" "test05_Massaging_AA")
case "$1" in
  "")
    set -x
    python -m isf.tests.unit_test -v
    ;;
  "s")
    set -x
    python -m isf.tests.unit_test -v TestStringMethods.test02_EqualizedOdds
    #python -m isf.tests.unit_test -v TestStringMethods.test04_RejectOptionClassification
    ;;
  "1"|"2"|"3"|"4"|"5")
    n=$(expr "$1" - 1)
    set -x
    python -m isf.tests.unit_test -v "${test_class}.${test_func[$n]}"
    ;;
esac
set +x
#python -m isf.tests.unit_test -v TestStringMethods.test01_AdversarialDebiasing
#python -m isf.tests.unit_test -v TestStringMethods.test02_EqualizedOdds
#python -m isf.tests.unit_test -v TestStringMethods.test03_Massaging
#python -m isf.tests.unit_test -v TestStringMethods.test04_RejectOptionClassification
#python -m isf.tests.unit_test -v TestStringMethods.test05_Massaging_AA
