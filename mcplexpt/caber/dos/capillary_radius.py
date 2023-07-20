# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:07:28 2022

@author: ttack
"""
#%% Tiff 설정
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mcplexpt.caber.dos import DoSCaBERExperiment
from mcplexpt.testing import get_samples_path
path = get_samples_path("caber", "dos","sample_250fps.tiff")
expt = DoSCaBERExperiment(path)

#%% 분석 명령어
'''엑셀파일만들기,원본파일,파일명'''
expt.Image_Radius_Measure('/home/minhyukim/multitif/211103_peo_0,05psi.tif','220208_PEO200_1_04')
#%%
'''엑셀파일로 분석'''
expt.Dos_CaBER_VE_analysis('220208_PEO200_1_04.xlsx',60)
#%%
'''Extensional_viscosity'''
expt.Extensional_viscosity_total('220208_PEO200_1_04.xlsx',60)

#%% 날짜Input Total 분석
'''날짜Input Total 분석,Surface tension '''
expt.Fast_Data_Analysis(220221,60)

# %%
