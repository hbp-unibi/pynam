#!/bin/sh

# Automatically run and discover unittests
python -m unittest discover || exit 1

# If pynnless is present in the lib/ folder, run its tests too
if [ -e lib/pynnless/test.sh ]; then
	(cd lib/pynnless; ./test.sh) || exit 1
fi
