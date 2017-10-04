#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:07:28 2017

@author: zyluo24
"""

# The total training number should be the total number of questios
# we need a dictionary that stoate the information like
# 1:image1_feature, 1: question, 2 :image1_feature 2: question2 for image1
# 3:image1_feature, 

# step1 construc the dictionary that contains theh information about  question

# {1: {'question': 'what is sheeating','image_id' :24356}}
# to find correponding img_feauture using image_id to find will be possible

import numpy as np
import json

dict1 = json.load(open("image_id_to_indexes.txt"))
dict2 = {}

i = 0
for keys in dict1.keys():
    List1 = dict1[keys]
    
    for indexes in List1:
        dict3 = {}
        dict3['question'] = indexes
        dict3['image_id'] = keys
        dict2[i] = dict3
        i +=1
print dict2
print i
with open('question_mapping_index.txt', 'w') as file:
    file.write(json.dumps(dict2))
    
#dict1 = json.load(open("fc7_feature_dict.txt"))





















    