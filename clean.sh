#!/bin/sh

find -depth \( \( -name "*.aux" -o -name "*.snm" -o -name "*.out" -o -name "*.toc" -o -name "*.nav" -o -name "*.log" -o -name "*.backup" -o -name "*.bbl" -o -name "*.blg" -o -name "*~" -o -name "CMakeCache.txt" -o -name "*.pyc" \) -a \! \( -wholename "*.git/*" -o -wholename "*.svn" \) \) -delete -print
