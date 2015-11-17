#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""
Contains various utility methods used in various places of the program.
"""

import re
import json
import numpy as np
import scipy.io as scio

def initialize_seed(seed, seq=1):
    """
    Initializes the numpy random number generator seed with the given seed
    value. The seed value may be None, in which case no changes the the numpy
    seed are made. Returns the old random generator state or None if no change
    was made.
    """
    if seed is None:
        return None
    old_state = np.random.get_state()
    np.random.seed(int(long(seed * (seq + 1)) % long(1 << 30)))
    return old_state

def finalize_seed(old_state):
    """
    Restores the numpy random seed to its old value, or does nothin if the given
    value is "None".
    """
    if (old_state != None):
        np.random.set_state(old_state)

# Regular expression for comments
# See http://www.lifl.fr/~damien.riquet/parse-a-json-file-with-comments.html
JSON_COMMENT_RE = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

def parse_json_with_comments(stream):
    """
    Parse a JSON stream, allow JavaScript like comments. Adapted from
    http://www.lifl.fr/~damien.riquet/parse-a-json-file-with-comments.html
    """
    content = ''.join(stream.readlines())
    match = JSON_COMMENT_RE.search(content)
    while match:
        content = content[:match.start()] + content[match.end():]
        match = JSON_COMMENT_RE.search(content)
    return json.loads(content)

# loadmat with python dictionaries
# See http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

def loadmat(filename):
    '''
    this function should be called instead of direct scio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# Try to forward the "init_key" method from pynnless
try:
    import pynnless.pynnless_utils
    init_key = pynnless.pynnless_utils.init_key
except:
    pass
