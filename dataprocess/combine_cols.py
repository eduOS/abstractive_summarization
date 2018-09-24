# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import pymongo
from utils import debug_line


def combine_cols():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    mycol = mydb["bytecup"]

    mycol_95 = mydb["95"]
    mycol_96 = mydb["96"]
    mycol_160 = mydb["160"]
    mycol_243 = mydb["243"]

    for mycol in [mycol_95, mycol_96, mycol_160, mycol_243]:
        g = mycol.find({"pos_tags": {"$exists": True}})
        rst = g.next()

        try:
            mycol.insert_one({
                        "_id": rst['_id'],
                        "old_id": rst["old_id"],
                        "new_id": rst["new_id"],
                        "orig_title": rst["orig_title"],
                        "orig_content": rst["orig_content"],
                        "content_sents": rst["content_sents"],
                        "pos_tag_words": rst["pos_tag_words"],
                        "pos_tags": rst["pos_tags"],
                        "ner_tags": rst["ner_tags"],
                        "stem": rst["stem"],
                        "tfidf_scores": rst["tfidf_scores"],
                        "phrase_indices": rst["phrase_indices"],
                        "sent_indices": rst["sent_indices"]}
            )
        except KeyError as e:
            debug_line("keyerror: not complete entry", str(rst))
            try:
                mycol.insert_one({
                    "_id": rst['_id'],
                    "old_id": rst["old_id"],
                    "new_id": rst["new_id"],
                    "orig_title": rst["orig_title"],
                    "orig_content": rst["orig_content"],
                    "content_sents": rst["content_sents"],
                })
            except:
                debug_line("%s(old id) lost" % rst["old_id"], str(rst))
        except Exception as e:
            debug_line(str(e), str(rst))
            try:
                mycol.insert_one({
                    "_id": rst['_id'],
                    "old_id": rst["old_id"],
                    "new_id": rst["new_id"],
                    "orig_title": rst["orig_title"],
                    "orig_content": rst["orig_content"],
                    "content_sents": rst["content_sents"],
                })
            except:
                debug_line("%s(old id) lost" % rst["old_id"], str(rst))

    mycol.close()
    mycol_95.close()
    mycol_96.close()
    mycol_160.close()
    mycol_243.close()


combine_cols()
