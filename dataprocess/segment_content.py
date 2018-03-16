# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import re

"""
1. pairs with less than 30% of word overlap should be deleted
    ["②", "①", "⑤", "⑥", "⑦", "⑧", "③", "④"]
"""


def is_delimiter(char, next_char=None):
    DELIMITERS = ["。", "：", ":", "∶", "？", "；", "！", "…",
                  "—", "?", ";", "!", "|", ".", "．"]
    if char in DELIMITERS or re.match(r'(\d+\.|\.\.+)', char):
        return True

    if not next_char:
        return False

    if char in ['～', '~'] and re.match("[^\d]", next_char[0]):
        return True
    return False


