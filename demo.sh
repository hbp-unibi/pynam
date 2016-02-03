#!/bin/bash
#   PyNAM -- Python Neural Associative Memory Simulator and Evaluator
#   Copyright (C) 2015 Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

#
# Performs a test run and plots the results
#

# Example usage:
# ./demo.sh # Run with spikey
# ./demo.sh nest # Run with nest
# SDK_PATH=/vol/misc/spikey_demo ./demo.sh # Run with spikey and given SDK path


SIMULATOR=${1:-spikey}
SDK_PATH=${SDK_PATH:-~/spikey}
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
	./run.py "$SIMULATOR" experiments/demo_2d_sweep_spikey.json 2>&1 | tee "$TMPFILE"
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
	evince out/plot_*_2d_spikey.pdf
fi


