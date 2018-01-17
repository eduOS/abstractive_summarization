# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from utils import sourceline2words
from codecs import open
import sys

with open("../../data/LCSTS/PART_III.txt", 'r', 'utf-8') as f_in, open(sys.argv[-1], 'w', 'utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        if line.startswith("<"):
            continue
        new_line = " ".join(sourceline2words(line))
        f_out.write(line + "\n")
        f_out.write(new_line + "\n\n")
