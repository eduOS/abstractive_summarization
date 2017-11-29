# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import os


def ensure_exists(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
