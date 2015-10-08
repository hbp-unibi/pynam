# -----------------------------------------------------------------------------
# CITEC - Center of Excellence Cognitive Interaction Technology
# Bielefeld University
# Cognitronics & Sensor Systems
#
# File Name    : log.py
# Author       : Andreas Stoeckel
# Description  : Module used for printing error and other log messages.

import __main__
import os
import sys

# Output stream to use
logstream = sys.stderr

def setLogStream(stream):
    global logstream
    logstream = stream

def getLogStream():
    global logstream
    return logstream

def printLogMessage(head, colorCode, *msg):
    useColor = os.isatty(logstream.fileno())
    if (useColor):
        logstream.write('\x1b[' + str(colorCode) + ';1m' + head + ":\x1b[0m ")
    else:
        logstream.write(head + ": ")
    if (hasattr(__main__, "__file__")):
        logstream.write('[' + os.path.basename(__main__.__file__) + '] ')
    logstream.write(' '.join(map(str, msg)) + "\n")
    logstream.flush()

# Logs a message as "note"
def note(*msg):
    printLogMessage('Note', 34, *msg)

# Logs a message as "warning"
def warning(*msg):
    printLogMessage('Warning', 35, *msg)

# Logs a message as "error"
def error(*msg):
    printLogMessage('Error', 31, *msg)

