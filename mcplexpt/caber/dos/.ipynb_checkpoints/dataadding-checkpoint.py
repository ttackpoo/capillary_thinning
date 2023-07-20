# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:20:09 2021

@author: ttack
"""

#%% json → dictionary 확인법
import json 

with open("C:/Users/ttack/mcplexpt/mcplexpt/mcplexpt/caber/dos/solutionpropertydata.json","r") as f:
 sample=json.load(f)   
 print(sample)
 
 ''' parameter 불러오기 : dictionary indexing
     ex) object = sample['211130_05_106_aj']['zsv']
    
'''
   