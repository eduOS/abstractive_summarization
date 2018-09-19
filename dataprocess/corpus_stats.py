# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import glob
import time
from codecs import open
from utils import read_origin, bytes2unicode, prep_cut_sent, load_json, debug_line
import pymongo

data_path = "./data/*.txt_*"
filelist = glob.glob(data_path)

illegal_num = 0
len_art = []
len_abs = []

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
mycol = mydb["bytecup2018"]


def analyze(infile, is_debug=0):
    fi = open(infile, 'rb')
    st = time.time()
    title_illegal_file = open('title_illegal_log', 'w', 'utf-8')
    json_illegal_file = open('json_illegal_log', 'w', 'utf-8')
    illegal_length_file = open('illegal_length_log', 'w', 'utf-8')
    log_file = open('corpus_log', 'a', 'utf-8')
    pas_len = []
    new_id = 0
    lines = 0
    sent_len = []
    pas_sen_len = []
    title_len = []
    json_illegal_num = 0
    title_illegal_num = 0
    illegal_length = 0
    duplicated_illegal_num = 0
    duplicated_file = open('duplicated_illegal_log', 'w', 'utf-8')

    while(True):
        ori_line = read_origin(fi, is_debug=0)
        lines += 1
        if not ori_line:
            json_illegal_file.write("\n\n" + str(json_illegal_num))
            title_illegal_file.write("\n\n" + str(title_illegal_num))
            duplicated_file.write("\n\n" + str(duplicated_illegal_num))
            illegal_length_file.write("\n\n" + str(illegal_length))
            illegal_length_file.close()
            json_illegal_file.close()
            title_illegal_file.close()
            duplicated_file.close()
            break

        line = bytes2unicode(ori_line)
        try:
            old_id, title, content = load_json(line)
            if new_id == 0 and title is None:
                new_id = 10 ** 8
        except:
            json_illegal_num += 1
            json_illegal_file.write(str(ori_line.decode('utf-8')))
            json_illegal_file.write('\n+++++\n\n')
            json_illegal_file.flush()
            continue

        if title is not None and 'u_n_k' in title:
            title_illegal_file.write(ori_line.decode('utf-8'))
            title_illegal_file.write('\n+++++\n\n')
            title_illegal_num += 1
            title_illegal_file.flush()
            continue

        pas_len.append(len(content.split()))
        if title is not None:
            title_len.append(len(title.split()))
        sents = prep_cut_sent(content, is_debug=0, log_time=0)
        if not sents or not title:
            continue
        sent_len.extend(list(map(lambda x: len(x.split()), sents)))
        pas_sen_len.append(len(sents))
        content_sents = "\n".join(sents)
        if len(content_sents) < len(content) / 2:
            illegal_length_file.write("\n\n---" + str(old_id) + "---" + title + "\n" + content + "\n" + content_sents + "------\n\n")
            illegal_length += 1
            continue
        # debug_line('content_sents', content_sents)
        if title is None:
            _title = "None"
        else:
            _title = title
        _id = hash(_title + content)
        try:
            mycol.insert_one(
                {
                    "_id": _id,
                    "old_id": old_id,
                    "new_id": new_id,
                    "orig_title": title,
                    "orig_content": content,
                    "content_sents": content_sents,
                }
            )
            new_id += 1
        except:
            dup = list(mycol.find({"_id": _id}))[0]
            _title = dup["orig_title"]
            _content = dup["orig_content"]
            if _title == title or _content == content:
                duplicated_file.write(ori_line.decode('utf-8'))
                duplicated_file.write('\n+++++\n\n')
                duplicated_illegal_num += 1
            else:
                duplicated_file.write("\n-----"+old_id+"\n-----")
                dup = mycol.find_one({"_id": _id})
                print(dup['old_id'])
                print(old_id)
                print(title)
                print(dup['orig_title'])
                print(content_sents)
                print(dup['content_sents'])
                input()

    log_file.write("the mean of passage length: %s" % float(np.mean(pas_len)))
    log_file.write("\n")
    log_file.write("the std of passage length: %s" % float(np.std(pas_len)))
    log_file.write("\n")
    log_file.write("the max of passage length: %s" % float(np.max(pas_len)))
    log_file.write("\n")
    log_file.write("the min of passage length: %s" % float(np.min(pas_len)))
    log_file.write("\n")
    log_file.write("\n")

    log_file.write("the mean of title length: %s" % float(np.mean(title_len)))
    log_file.write("\n")
    log_file.write("the std of title length: %s" % float(np.std(title_len)))
    log_file.write("\n")
    log_file.write("the max of title length: %s" % float(np.max(title_len)))
    log_file.write("\n")
    log_file.write("the min of title length: %s" % float(np.min(pas_len)))
    log_file.write("\n")
    log_file.write("\n")

    log_file.write("the mean of passage sent_len: %s" % float(np.mean(pas_sen_len)))
    log_file.write("\n")
    log_file.write("the std of passage sent_len: %s" % float(np.std(pas_sen_len)))
    log_file.write("\n")
    log_file.write("the max of passage sent_len: %s" % float(np.max(pas_sen_len)))
    log_file.write("\n")
    log_file.write("the min of passage sent_len: %s" % float(np.min(pas_sen_len)))
    log_file.write("\n")
    log_file.write("\n")

    log_file.write("the mean of sent_len: %s" % float(np.mean(sent_len)))
    log_file.write("\n")
    log_file.write("the std of sent len: %s" % float(np.std(sent_len)))
    log_file.write("\n")
    log_file.write("the max of sent_len: %s" % float(np.max(sent_len)))
    log_file.write("\n")
    log_file.write("the min of sent_len: %s" % float(np.min(sent_len)))
    log_file.write("\n")
    log_file.write("\n")
    log_file.write("everage time: %2.2f ms\n" % ((time.time()-st) * 1000/lines))
    log_file.write("total lines %s\n" % lines)
    log_file.write("unduplicated lines %s\n" % (lines-duplicated_illegal_num))
    log_file.write("illegal length %s\n" % (illegal_length))
    # 709540

    log_file.close()

if __name__ == '__main__':
    in_file = './data/bytecup.corpus.train.txt'
    analyze(in_file, is_debug=0)
