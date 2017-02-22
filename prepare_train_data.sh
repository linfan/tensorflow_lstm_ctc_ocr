#!/usr/bin/env bash

wget "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"

python extractbgs.py SUN397.tar.gz
