# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import codecs
import glob
import json
import cntk

data_path_regex = '~/data/bytecup2018/corpus/bytecup.corpus*'
filelist = glob.glob(data_path_regex)  # get the list of datafiles


class Processor():
    """
    dirty contents:
        1) that can be converted to asscii should be converted to asscii
        2)
        3)
    """


def convert_punc(text):

    return

for file in filelist:
    for line in codecs.open(file, 'r'):
        item = json.loads(line)
        content = item['content']
        title = item['title']
