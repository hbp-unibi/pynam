#!/bin/bash

#
# Performs a test run and plots the results
#

# Example usage:
# ./demo.sh # Run with spikey
# ./demo.sh nest # Run with nest
# SDK_PATH=/vol/misc/spikey_demo ./demo.sh # Run with spikey and given SDK path


SIMULATOR=${1:-spikey}
SDK_PATH=${SDK_PATH:-~/source/spikey_demo}
TMPFILE=`mktemp`

(
	# Load the Spikey environment
	if [ "$SIMULATOR" = "spikey" ]; then
		if [ ! -d "$SDK_PATH" ]; then
			echo "$SDK_PATH not found, please set the SDK_PATH environment"
			     "variable to the correct spikey SDK path"
			exit 1
		fi
		source "$SDK_PATH"/bin/env.sh
	fi

	# Execute the PyNAM main script
	./run.py "$SIMULATOR" 2>&1 | tee "$TMPFILE"
)

# Plot the created output
TARGET=`cat "$TMPFILE"\
		| grep "Writing target file: "\
		| head -n 1\
		| sed 's/Writing target file: \(\)/\1/'`
TARGET=`readlink -e "$TARGET"`
echo $TARGET
rm "$TMPFILE"
if [ -e "$TARGET" ]; then
	cd ./misc
	./information_plot.py "$TARGET"
	evince out/plot_*.pdf
fi


