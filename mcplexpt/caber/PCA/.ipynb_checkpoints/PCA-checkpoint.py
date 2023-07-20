# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:38:58 2022

@author: ttack
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:00:35 2022
For example, the model predicts class 4, which is a mixed fluid, as class 3, which is a CMC single fluid, and this difference is attributed to the lack of information due to the nature of the two fluids to break in a relatively short frame compared to other classes.
@author: MCPL-JJ
"""


import cv2
import numpy as np
import os
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mcplexpt.caber.dos import DoSCaBERExperiment
from mcplexpt.testing import get_samples_path
from sklearn.preprocessing import StandardScaler  ##StandardScaler :  평균0,분산1로 표준화 해준다.
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import json
import pandas as pd
from numpy import random
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error




class pcaclass:

      
   
    
    def eigen4(a,b,c,d,pca_c=10):
    
        '''
        
        
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        c : str
            분류유체3의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
        
        Returns
        -------
        None.
        
            """  
    >>> from PCA import pcaclass


        """

        '''
        pca_c=10
        a='CMC_crop2'
        b='PEO_crop2'
        c='Carbopol_crop2'
        d='PEO3,CMC3,Carbopol2_crop2'
        base_dir = 'C:/Users/ttack/PCA'
        CMC_dir = os.path.join(base_dir,'{}'.format(a))
        PEO_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        
        CMC_list = os.listdir(CMC_dir)
        PEO_list = os.listdir(PEO_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        
        
        
        CMC_images = np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0),axis=0)
        PEO_images = np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_images = np.concatenate((CMC_images,np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_images = np.concatenate((PEO_images,np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        images = np.concatenate((CMC_images,PEO_images,c_images,d_images),axis=0)
        
        CMC_data = np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0).flatten()),axis=0)
        PEO_data = np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_data = np.concatenate((CMC_data,np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_data = np.concatenate((PEO_data,np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))    
        data = np.concatenate((CMC_data,PEO_data,c_data,d_data),axis=0)    
        
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(CMC_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(PEO_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target = target_0 + target_1 +target_2 + target_3##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c),'{}'.format(d)]
        
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
        
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 ##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        # '''
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        # '''
        X_people = data
        y_people = target 
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
        
        
        test_image = cv2.imread('C:/Users/ttack/PCA/220622_class1/220525_Carbo5PEO5_2skip_01multi_image.png',0)
        test_image = test_image.reshape(1,-1)
        proba_list = []
        # ##향후에 k값 변경 필요시 사용
        # k_list = range(1,10)
        # accuracies = []
        # for k in k_list:
        #   classifier = KNeighborsClassifier(n_neighbors = k)
        #   classifier.fit( X_train, y_train)
        #   accuracies.append(classifier.score(X_test, y_test))
        #   proba = classifier.predict_proba(test_image)
        #   proba_list.append(proba)
          
        # plt.plot(k_list, accuracies)
        # plt.xlabel("k")
        # plt.ylabel("Validation Accuracy")
        # plt.title("Fluid Classifier Accuracy")
        # plt.show()
        
        
        
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(X_train, y_train)
        print("{:2f}".format(knn.score(X_test, y_test)))
        
        ##
        pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("X_train_pca.shape: {}".format(X_train_pca.shape))
        
        knn_p = KNeighborsClassifier(n_neighbors=4)
        knn_p.fit(X_train_pca, y_train)
        print("{:.2f}".format(knn_p.score(X_test_pca, y_test)))
        
        ##
        fig, axes = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': (), 'yticks': ()})
        for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
            ax.imshow(component.reshape(image_shape), cmap='viridis')
            ax.set_title("PC {}".format(i+1))
    
        return knn, pca, knn_p
    

    
    def eigen4_TEST(a,b,c,pca_c=10):
        '''
        
    
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        c : str
            분류유체3의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
    
        Returns
        -------
        None.
    
        '''
        base_dir = 'C:/Users/ttack/PCA'
        CMC_dir = os.path.join(base_dir,'{}'.format(a))
        PEO_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        
        CMC_list = os.listdir(CMC_dir)
        PEO_list = os.listdir(PEO_dir)
        c_list = os.listdir(c_dir)
    
        
        CMC_images = np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0),axis=0)
        PEO_images = np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_images = np.concatenate((CMC_images,np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_images = np.concatenate((PEO_images,np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        images = np.concatenate((CMC_images,PEO_images,c_images),axis=0)
        
        CMC_data = np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0).flatten()),axis=0)
        PEO_data = np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_data = np.concatenate((CMC_data,np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_data = np.concatenate((PEO_data,np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0)))    
        data = np.concatenate((CMC_data,PEO_data,c_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(CMC_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(PEO_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target = target_0 + target_1 +target_2##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 ##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        # '''
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
      
        # X_people = X_people / 255.
        # '''
        X_people = data
        y_people = target 
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
        
        
        test_image = cv2.imread('C:/Users/ttack/PCA/220622_class1/220525_Carbo5PEO5_2skip_01multi_image.png',0)
        test_image = test_image.reshape(1,-1)
        proba_list = []
        # ##향후에 k값 변경 필요시 사용
        # k_list = range(1,10)
        # accuracies = []
        # for k in k_list:
        #   classifier = KNeighborsClassifier(n_neighbors = k)
        #   classifier.fit( X_train, y_train)
        #   accuracies.append(classifier.score(X_test, y_test))
        #   proba = classifier.predict_proba(test_image)
        #   proba_list.append(proba)
          
        # plt.plot(k_list, accuracies)
        # plt.xlabel("k")
        # plt.ylabel("Validation Accuracy")
        # plt.title("Fluid Classifier Accuracy")
        # plt.show()
       
        
       
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(X_train, y_train)
        print("{:2f}".format(knn.score(X_test, y_test)))
        
        ##
        pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("X_train_pca.shape: {}".format(X_train_pca.shape))
        
        knn_p = KNeighborsClassifier(n_neighbors=4)
        knn_p.fit(X_train_pca, y_train)
        print("{:.2f}".format(knn_p.score(X_test_pca, y_test)))
        
        ##
        fig, axes = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': (), 'yticks': ()})
        for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
            ax.imshow(component.reshape(image_shape), cmap='viridis')
            ax.set_title("PC {}".format(i+1))
        
        
        return knn, pca, knn_p
    
    

    
    def eigen2(a,b,pca_c=10):
        '''
        
    
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
    
        Returns
        -------
        None.
    
        '''
        base_dir = 'C:/Users/MCPL-JJ/eigen'
        CMC_dir = os.path.join(base_dir,'{}'.format(a))
        PEO_dir = os.path.join(base_dir,'{}'.format(b))
        
        CMC_list = os.listdir(CMC_dir)
        PEO_list = os.listdir(PEO_dir)
        
        CMC_images = np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0),axis=0)
        PEO_images = np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0),axis=0)
       
        for i in range(1,len(CMC_list)):
            CMC_images = np.concatenate((CMC_images,np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_images = np.concatenate((PEO_images,np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0),axis=0)))
     
        images = np.concatenate((CMC_images,PEO_images),axis=0)
        
        CMC_data = np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0).flatten()),axis=0)
        PEO_data = np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0).flatten()),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_data = np.concatenate((CMC_data,np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_data = np.concatenate((PEO_data,np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0).flatten()),axis=0)))
      
        data = np.concatenate((CMC_data,PEO_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(CMC_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(PEO_list)):
            target_1.append(1)
    
        target = target_0 + target_1##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 ##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        # '''
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
      
        # X_people = X_people / 255.
        # '''
        X_people = data
        y_people = target 
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
        
        
        test_image = cv2.imread('C:/Users/MCPL-JJ/eigen/new_test_image/220525_Carbo5PEO5_2skip_13multi_image.png',0)
        test_image = test_image.reshape(1,-1)
        proba_list = []
        # ##향후에 k값 변경 필요시 사용
        # k_list = range(1,10)
        # accuracies = []
        # for k in k_list:
        #   classifier = KNeighborsClassifier(n_neighbors = k)
        #   classifier.fit( X_train, y_train)
        #   accuracies.append(classifier.score(X_test, y_test))
        #   proba = classifier.predict_proba(test_image)
        #   proba_list.append(proba)
          
        # plt.plot(k_list, accuracies)
        # plt.xlabel("k")
        # plt.ylabel("Validation Accuracy")
        # plt.title("Fluid Classifier Accuracy")
        # plt.show()
       
        
       
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(X_train, y_train)
        print("{:2f}".format(knn.score(X_test, y_test)))
        
        ##
        pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("X_train_pca.shape: {}".format(X_train_pca.shape))
        
        knn_p = KNeighborsClassifier(n_neighbors=4)
        knn_p.fit(X_train_pca, y_train)
        print("{:.2f}".format(knn_p.score(X_test_pca, y_test)))
        
        ##
        fig, axes = plt.subplots(3,5, figsize=(15,8), subplot_kw={'xticks': (), 'yticks': ()})
        for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
            ax.imshow(component.reshape(image_shape), cmap='viridis')
            ax.set_title("PC {}".format(i+1))
        
        
        return knn, pca, knn_p

    
    def eigen5(a,b,c,d,e,pca_c=10):
        '''
        
    
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        c : str
            분류유체3의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
    
        Returns
        -------
        None.
    
        '''
        base_dir = 'C:/Users/MCPL-JJ/eigen'
        CMC_dir = os.path.join(base_dir,'{}'.format(a))
        PEO_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        e_dir = os.path.join(base_dir,'{}'.format(e))
        
        CMC_list = os.listdir(CMC_dir)
        PEO_list = os.listdir(PEO_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        e_list = os.listdir(e_dir)
    
        
        CMC_images = np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0),axis=0)
        PEO_images = np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        e_images = np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_images = np.concatenate((CMC_images,np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_images = np.concatenate((PEO_images,np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        for i in range(1,len(e_list)):
            e_images = np.concatenate((e_images,np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0),axis=0)))
        images = np.concatenate((CMC_images,PEO_images,c_images,d_images,e_images),axis=0)
        
        CMC_data = np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0).flatten()),axis=0)
        PEO_data = np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        e_data = np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0).flatten()),axis=0)
        for i in range(1,len(CMC_list)):
            CMC_data = np.concatenate((CMC_data,np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_data = np.concatenate((PEO_data,np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(e_list)):
            e_data = np.concatenate((e_data,np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((CMC_data,PEO_data,c_data,d_data,e_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(CMC_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(PEO_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target_4 = []
        for i in range(0,len(e_list)):
            target_4.append(4)
        target = target_0 + target_1 +target_2 + target_3 + target_4##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 + target_4##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        X_people = data
        y_people = target
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
        
        '''
        ##향후에 k값 변경 필요시 사용
        k_list = range(1,101)
        accuracies = []
        for k in k_list:
          classifier = KNeighborsClassifier(n_neighbors = k)
          classifier.fit(training_data, training_labels)
          accuracies.append(classifier.score(validation_data, validation_labels))
        plt.plot(k_list, accuracies)
        plt.xlabel("k")
        plt.ylabel("Validation Accuracy")
        plt.title("Breast Cancer Classifier Accuracy")
        plt.show()
        ''' 
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        print("{:2f}".format(knn.score(X_test, y_test)))
        
        ##
        pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("X_train_pca.shape: {}".format(X_train_pca.shape))
        
        knn_p = KNeighborsClassifier(n_neighbors=1)
        knn_p.fit(X_train_pca, y_train)
        print("{:.2f}".format(knn_p.score(X_test_pca, y_test)))
        
        ##
        fig, axes = plt.subplots(3,5, figsize=(15,12), subplot_kw={'xticks': (), 'yticks': ()})
        for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
            ax.imshow(component.reshape(image_shape), cmap='viridis')
            ax.set_title("PC {}".format(i+1))
        
        
        return knn, pca, knn_p
    
    def component(self,a,b,c,d,e,f,g,pca_c):

        filename=a[7:]+b[7:]+c[7:]+d[7:]+e[7:]+f[7:]+g[7:]
        base_dir = '/home/minhyukim/PCA/'
        save_dir = base_dir+'PCA_result/'+filename

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)


        a_dir = os.path.join(base_dir,'{}'.format(a))
        b_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        e_dir = os.path.join(base_dir,'{}'.format(e))
        f_dir = os.path.join(base_dir,'{}'.format(f))
        g_dir = os.path.join(base_dir,'{}'.format(g))
        
        a_list = os.listdir(a_dir)
        b_list = os.listdir(b_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        e_list = os.listdir(e_dir)
        f_list = os.listdir(f_dir)
        g_list = os.listdir(g_dir)
        
        a_images = np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0),axis=0)
        b_images = np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        e_images = np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0),axis=0)
        f_images = np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0),axis=0)
        g_images = np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0),axis=0)
        for i in range(1,len(a_list)):
            a_images = np.concatenate((a_images,np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0),axis=0)))
        for i in range(1,len(b_list)):
            b_images = np.concatenate((b_images,np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        for i in range(1,len(e_list)):
            e_images = np.concatenate((e_images,np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0),axis=0)))
        for i in range(1,len(f_list)):
            f_images = np.concatenate((f_images,np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0),axis=0)))
        for i in range(1,len(g_list)):
            g_images = np.concatenate((g_images,np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0),axis=0)))
        images = np.concatenate((a_images,b_images,c_images,d_images,e_images,f_images,g_images),axis=0)
        
        a_data = np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0).flatten()),axis=0)
        b_data = np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        e_data = np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0).flatten()),axis=0)
        f_data = np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0).flatten()),axis=0)
        g_data = np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0).flatten()),axis=0)
        for i in range(1,len(a_list)):
            a_data = np.concatenate((a_data,np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(b_list)):
            b_data = np.concatenate((b_data,np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(e_list)):
            e_data = np.concatenate((e_data,np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(f_list)):
            f_data = np.concatenate((f_data,np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(g_list)):
            g_data = np.concatenate((g_data,np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((a_data,b_data,c_data,d_data,e_data,f_data,g_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(a_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(b_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target_4 = []
        for i in range(0,len(e_list)):
            target_4.append(4)
        target_5 = []
        for i in range(0,len(f_list)):
            target_5.append(5)
        target_6 = []
        for i in range(0,len(g_list)):
            target_6.append(6)
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d),'{}'.format(e), '{}'.format(f),'{}'.format(g)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        '''fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()'''     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        X_people = data
        y_people = target
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
        
        '''
        ##향후에 k값 변경 필요시 사용
        k_list = range(1,101)
        accuracies = []
        for k in k_list:
          classifier = KNeighborsClassifier(n_neighbors = k)
          classifier.fit(training_data, training_labels)
          accuracies.append(classifier.score(validation_data, validation_labels))
        plt.plot(k_list, accuracies)
        plt.xlabel("k")
        plt.ylabel("Validation Accuracy")
        plt.title("Breast Cancer Classifier Accuracy")
        plt.show()
        ''' 
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        '''scaler= StandardScaler()
        X_people = scaler.fit(data)
        StandardScaler()
        X_people = scaler.transform(data)'''        
        pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_people)        
        fig, axes = plt.subplots(3,5, figsize=(15,12), subplot_kw={'xticks': (), 'yticks': ()},dpi=300)      
        for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):            
            ax.imshow(component.reshape(image_shape), cmap='viridis')            
            ax.set_title("PC {}".format(i+1))
            
        datavar=np.var(X_people,0)
        varmax=np.max(datavar) 
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_people_pca = pca.transform(X_people)
        components=pca.components_
        variance=pca.explained_variance_
        mean=pca.mean_        

        list_list=[a_list,b_list,c_list,d_list,e_list,f_list,g_list]
        scoreaverage=[]
        scorevariance=[]        
       
        for j in range(0,pca_c):
            averagelist=[]
            variancelist=[]
            flag=0
            for i in range(0,7):
                if i == 0:
                    average=sum(X_people_pca[0:len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    variance=np.var(X_people_pca[0:len(list_list[i]),j],axis=0)
                    variancelist.append(variance)
                    flag=len(list_list[i])                    
                else:                    
                    average=sum(X_people_pca[flag:flag+len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    variance=np.var(X_people_pca[flag:flag+len(list_list[i]),j],axis=0)
                    variancelist.append(variance)
                    flag=flag+len(list_list[i])
                    
            scoreaverage.append(averagelist)
            scorevariance.append(variancelist)



        return components, scoreaverage, image_shape, variance, mean, X_people, scorevariance, y_people, X_people_pca
        
    
    
    
         
    def eigen7(self,a,b,c,d,e,f,g,pc1,pc2,pc3,stackup_pc,pcpxindex,pca_c,pixel):
        '''
        
    
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        c : str
            분류유체3의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
    
        Returns
        -------
        None.
    
        '''
        '''
        filename=a[7:]+b[7:]+c[7:]+d[7:]+e[7:]+f[7:]+g[7:]
        base_dir = '/home/minhyukim/PCA/'
        save_dir = base_dir+'PCA_result/'+filename

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)


        a_dir = os.path.join(base_dir,'{}'.format(a))
        b_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        e_dir = os.path.join(base_dir,'{}'.format(e))
        f_dir = os.path.join(base_dir,'{}'.format(f))
        g_dir = os.path.join(base_dir,'{}'.format(g))
        
        a_list = os.listdir(a_dir)
        b_list = os.listdir(b_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        e_list = os.listdir(e_dir)
        f_list = os.listdir(f_dir)
        g_list = os.listdir(g_dir)
   
        
        a_images = np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0),axis=0)
        b_images = np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        e_images = np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0),axis=0)
        f_images = np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0),axis=0)
        g_images = np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0),axis=0)
        for i in range(1,len(a_list)):
            a_images = np.concatenate((a_images,np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0),axis=0)))
        for i in range(1,len(b_list)):
            b_images = np.concatenate((b_images,np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        for i in range(1,len(e_list)):
            e_images = np.concatenate((e_images,np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0),axis=0)))
        for i in range(1,len(f_list)):
            f_images = np.concatenate((f_images,np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0),axis=0)))
        for i in range(1,len(g_list)):
            g_images = np.concatenate((g_images,np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0),axis=0)))
        images = np.concatenate((a_images,b_images,c_images,d_images,e_images,f_images,g_images),axis=0)
        
        a_data = np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0).flatten()),axis=0)
        b_data = np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        e_data = np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0).flatten()),axis=0)
        f_data = np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0).flatten()),axis=0)
        g_data = np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0).flatten()),axis=0)
        for i in range(1,len(a_list)):
            a_data = np.concatenate((a_data,np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(b_list)):
            b_data = np.concatenate((b_data,np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(e_list)):
            e_data = np.concatenate((e_data,np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(f_list)):
            f_data = np.concatenate((f_data,np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(g_list)):
            g_data = np.concatenate((g_data,np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((a_data,b_data,c_data,d_data,e_data,f_data,g_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(a_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(b_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target_4 = []
        for i in range(0,len(e_list)):
            target_4.append(4)
        target_5 = []
        for i in range(0,len(f_list)):
            target_5.append(5)
        target_6 = []
        for i in range(0,len(g_list)):
            target_6.append(6)
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d),'{}'.format(e), '{}'.format(f),'{}'.format(g)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        X_people = data
        y_people = target
        '''
        filename=a[7:]+b[7:]+c[7:]+d[7:]+e[7:]+f[7:]+g[7:]
        base_dir = '/home/minhyukim/PCA/'
        save_dir = base_dir+'PCA_result/'+filename

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)


        a_dir = os.path.join(base_dir,'{}'.format(a))
        b_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        e_dir = os.path.join(base_dir,'{}'.format(e))
        f_dir = os.path.join(base_dir,'{}'.format(f))
        g_dir = os.path.join(base_dir,'{}'.format(g))

        a_list = os.listdir(a_dir)
        b_list = os.listdir(b_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        e_list = os.listdir(e_dir)
        f_list = os.listdir(f_dir)
        g_list = os.listdir(g_dir)


        a_images = np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0),axis=0)
        b_images = np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        e_images = np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0),axis=0)
        f_images = np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0),axis=0)
        g_images = np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0),axis=0)
        for i in range(1,len(a_list)):
            a_images = np.concatenate((a_images,np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0),axis=0)))
        for i in range(1,len(b_list)):
            b_images = np.concatenate((b_images,np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        for i in range(1,len(e_list)):
            e_images = np.concatenate((e_images,np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0),axis=0)))
        for i in range(1,len(f_list)):
            f_images = np.concatenate((f_images,np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0),axis=0)))
        for i in range(1,len(g_list)):
            g_images = np.concatenate((g_images,np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0),axis=0)))
        images = np.concatenate((a_images,b_images,c_images,d_images,e_images,f_images,g_images),axis=0)

        a_data = np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0).flatten()),axis=0)
        b_data = np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        e_data = np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0).flatten()),axis=0)
        f_data = np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0).flatten()),axis=0)
        g_data = np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0).flatten()),axis=0)
        for i in range(1,len(a_list)):
            a_data = np.concatenate((a_data,np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(b_list)):
            b_data = np.concatenate((b_data,np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(e_list)):
            e_data = np.concatenate((e_data,np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(f_list)):
            f_data = np.concatenate((f_data,np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(g_list)):
            g_data = np.concatenate((g_data,np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((a_data,b_data,c_data,d_data,e_data,f_data,g_data),axis=0)
        data1 = np.array(data)
        

        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(a_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(b_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target_4 = []
        for i in range(0,len(e_list)):
            target_4.append(4)
        target_5 = []
        for i in range(0,len(f_list)):
            target_5.append(5)
        target_6 = []
        for i in range(0,len(g_list)):
            target_6.append(6)
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)

        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d),'{}'.format(e), '{}'.format(f),'{}'.format(g)]

        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        '''fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()'''     

        ##
        X_people=data1
                
        y_people=target

        ##target 이 자꾸1로 변경되서 다시 추가함
       
        
        MSE_pc_list=[]           
        MSE_pc_list123=[]           
        MSE_pc_list4567=[]
        MSE_shuffle_list=[]
        
        for p in range(5,pca_c+1,10) :           
            
            MSE_shuffle_list=[]
            MSE_shuffle_list123=[]
            MSE_shuffle_list4567=[]
            for j in range(30):
                
                print(np.shape(X_people),np.shape(y_people))         
                X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, test_size=0.04,  shuffle=True)
                pca = PCA(n_components=p-1, whiten=True, random_state=0).fit(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                      
                '''
                ##향후에 k값 변경 필요시 사용
                k_list = range(1,101)
                accuracies = []
                for k in k_list:
                  classifier = KNeighborsClassifier(n_neighbors = k)
                  classifier.fit(training_data, training_labels)
                  accuracies.append(classifier.score(validation_data, validation_labels))
                plt.plot(k_list, accuracies)
                plt.xlabel("k")
                plt.ylabel("Validation Accuracy")
                plt.title("Breast Cancer Classifier Accuracy")
                plt.show()
                ''' 
                #knn = KNeighborsClassifier(n_neighbors=3)
                #knn.fit(X_train, y_train)

                
                
                datavar=np.var(X_people,0)
                varmax=np.max(datavar) 
                X_train_pca = pca.transform(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
        
                
                X_test_pca = pca.transform(X_test)
        
                X_people_pca = pca.transform(X_people)
        
                print(np.shape(X_train_pca),np.shape(X_test_pca),np.shape(X_people_pca))
                '''  
                proba_list = []        
                accuracies = []
                classifier = KNeighborsClassifier(n_neighbors = 10)
                for i in range(80) :        
                   pca = PCA(n_components=i+2, whiten=True, random_state=0).fit(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                   X_train_pca = pca.transform(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                   X_test_pca = pca.transform(X_test)
                   classifier.fit(X_train_pca, y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                   accuracies.append(classifier.score(X_test_pca, y_test))
                   proba = classifier.predict_proba(X_test_pca)
                   proba_list.append(proba)

                fig1,ax1= plt.subplots()
                ax1.plot(range(80),accuracies) 
                ax1.set_xlabel("components")
                ax1.set_ylabel("Validation Accuracy")
                ax1.axvline(x=16, color='red', linestyle='--')
                ax1.set_title("Fluid Classifier Accuracy")
                fig1.savefig('{}/{}'.format(save_dir,"Proportion"))'''


                #fig2, ax2 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
                #for i, (component, ax2) in enumerate(zip(pca.components_, ax2.ravel())):
                #    ax2.imshow(component.reshape(image_shape), cmap='viridis')
                #    ax2.set_title("PC {}".format(i+1))

                #fig2.savefig('{}/{}'.format(save_dir,"Principal component"))


                Fluid=['R1','P1','R2','P2','R3','P3','R4','P4','R5','P5','R6','P6','R7','P7']
                xticks=[0,0.4,1,1.4,2,2.4,3,3.4,4,4.4,5,5.4,6,6.4]
                Carboref=[1,0,0,0.5,0.5 ,0,    0.33]
                CMCref=  [0,1,0,0.5,0 ,  0.5,  0.33]
                PEOref=  [0,0,1,0  ,0.5 ,0.5,  0.33]
                matric=np.vstack([Carboref,CMCref,PEOref])
                


                #k_list = range(1,10)

                classifier = KNeighborsClassifier(n_neighbors=6)
                classifier.fit(X_train_pca, y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                #classifier.fit(X_train_pca, y_train)
                proba_list=[]
                
            
                proba = classifier.predict_proba(X_test_pca) 
                
                
                proba_list.append(proba)                

                #print(np.argmax(proba,axis=1))
                #print('proba',proba,'y_test',y_test)

                #f1=f1_score(y_test,np.argmax(proba,axis=1),average='macro')
                #f2=f1_score(y_test,np.argmax(proba,axis=1),average='micro')
                #f3=f1_score(y_test,np.argmax(proba,axis=1),average='weighted')


                #print('macro',f1)
                #print('micro',f2)
                #print('weighted',f3)

                Carbo_y=[[],[],[],[],[],[],[]]
                CMC_y=[[],[],[],[],[],[],[]]
                PEO_y=[[],[],[],[],[],[],[]]

                '''for i in range(len(y_test)):
                    concent=matric[:,np.where(proba[i]!=0)]*proba[i][np.where(proba[i]!=0)]
                    concent_sum=np.sum(concent,axis=-1)
                    Carbo_y[y_test[i]].append(concent_sum[0])
                    CMC_y[y_test[i]].append(concent_sum[1])
                    PEO_y[y_test[i]].append(concent_sum[2])
                    print('matric',matric[:,np.where(proba[i]!=0)],'proba',proba[i][np.where(proba[i]!=0)],'concent',concent,'y_test',y_test[i],'concent_sum',concent_sum)'''

            
                
                
                proba_list=proba_list[0]
                for k in range(len(proba_list)):
                    Carbo_y[y_test[k]].append(proba_list[k][0])
                   
                    CMC_y[y_test[k]].append(proba_list[k][1])
                    PEO_y[y_test[k]].append(proba_list[k][2])

                

                '''
                fig10,ax10 = plt.subplots()

                ax10.bar([0,1,2,3,4,5,6],Carboref,color='r',width=0.2)
                ax10.bar([0,1,2,3,4,5,6],CMCref,color='g', width=0.2,bottom=Carboref)
                ax10.bar([0,1,2,3,4,5,6],PEOref,color='b', width=0.2,bottom=[Carboref[i]+CMCref[i] for i in range(len(Carboref))])


                proba_list=np.array(proba_list)

                print(y_test)
                print('Carbo_y',Carbo_y)
                '''
                Carbo_y=[np.average(i) for i in Carbo_y]
          
                CMC_y=[np.average(i) for i in CMC_y]
                PEO_y=[np.average(i) for i in PEO_y]
                '''
                ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],Carbo_y,color='r',width=0.2,label='Carbopol')        
                ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],CMC_y,bottom=Carbo_y,color='g',width=0.2,label='CMC')
                ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],PEO_y,bottom=[Carbo_y[i]+CMC_y[i] for i in range(7)],color='b',width=0.2,label='PEO')
                ax10.set_title('Concentration ratio_{}_{}'.format(i,j))
                ax10.set_xticks(xticks)
                ax10.set_xticklabels(Fluid)
                ax10.legend()

                fig10.savefig('{}/{}'.format(save_dir,"Concentration ratio"))
                '''
                matric2=np.vstack([Carbo_y,CMC_y,PEO_y])
                MSE_list=[]
                for i in range(7):
                    #print('matric1',matric[:,i],'matric2',matric2[:,i])
                    MSE=mean_squared_error(matric[:,i],matric2[:,i])
                    MSE_list.append(MSE)
                MSE_average=np.average(MSE_list)
                #print('MSE_list',MSE_list)
                MSE_average123=np.average(MSE_list[0:3])
                MSE_average4567=np.average(MSE_list[3:-1])
                MSE_shuffle_list.append(MSE_average)
                MSE_shuffle_list123.append(MSE_average123)
                MSE_shuffle_list4567.append(MSE_average4567)
                print('min_random_state:',np.argmin(MSE_shuffle_list),'pc:',p)
                print('max_random_state:',np.argmax(MSE_shuffle_list),'pc:',p)
                
            MSE_pc_list.append(MSE_shuffle_list)
            MSE_pc_list123.append(MSE_shuffle_list123)
            MSE_pc_list4567.append(MSE_shuffle_list4567)
            #print('min_MSE',np.argmin(np.array(MSE_shuffle_list),axis=-1))
     
        
        fig1,ax1=plt.subplots(figsize=(12,12))
        ax1.boxplot(MSE_pc_list,[i for i in range(5,pca_c+1,5)])
        ax1.boxplot(MSE_pc_list123,[i for i in range(5,pca_c+1,5)])
        ax1.boxplot(MSE_pc_list4567,[i for i in range(5,pca_c+1,5)])
        fig2,ax2=plt.subplots(figsize=(12,12))

        for i in range(5,pca_c+1,10):            
            ax2.scatter(i*np.ones_like(MSE_pc_list123[int(i/10-1)]),MSE_pc_list123[int(i/10-1)], color='r', label='123')
        for i in range(5,pca_c+1,10):            
            ax2.scatter(i*np.ones_like(MSE_pc_list4567[int(i/10-1)]),MSE_pc_list4567[int(i/10-1)], color='g', label='4567')     
            
        ax2.set_xticks(range(5,pca_c+1,10))
        ax2.set_xlabel('PC')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.set_title("MSE for {}_frame weight".format(pixel))
        '''
#Optimal Truncation!!
        n_samples,n_features=X_train.shape
        singular_values=pca.singular_values_
        print(singular_values)
        mpmax=(1+np.sqrt(n_samples/n_features))**2
        mpmin=(1-np.sqrt(n_samples/n_features))**2
        mpmed=(mpmax+mpmin)/2
        print('mpmed',mpmed)
        threshold=4/(3**1/2)/np.sqrt(mpmed)*np.median(singular_values)
        
        print('threshold',threshold)
        components=pca.components_
       

        print('length',np.where(singular_values>threshold))
           
        
        

        variance_ratio_cumsum=np.cumsum(pca.explained_variance_ratio_)  
        
        fig13,ax13=plt.subplots()
        ax13.plot(variance_ratio_cumsum)
        ax13.set_xlabel('Number of PC')
        ax13.set_ylabel('Explained variance')
        
   
        fig14,ax14=plt.subplots()
        ax14.semilogy([i for i in range(len(singular_values))],singular_values)
        ax14.axhline(y=singular_values[-1], color='red',linestyle='--')
        ax14.text(0, singular_values[-1],"{:.2f}".format(singular_values[-1]),color='red')
        
        ax14.set_ylabel('Variance')
        
#Marcenko-Pastur 분포

        lambda_values = np.linspace(mpmin, mpmax, 500)
        alpha = n_samples/n_features
        # Marcenko-Pastur 분포 계산
        pdf_values = n_features/(2*np.pi*n_samples*alpha) * np.sqrt((mpmax - lambda_values)*(lambda_values - mpmin)) / lambda_values

        # 그래프 그리기
        fig14,ax14=plt.subplots()
        ax14.plot(lambda_values, pdf_values)
        ax14.set_title("Marcenko-Pastur Distribution (n_samples={}, n_features={}, alpha={:.2f})".format(n_samples, n_features, alpha))
        ax14.set_xlabel("Eigenvalue")
        ax14.set_ylabel("Density")
        '''
        

            

      
    
    
#3D SCATTER            
       
        
                
        '''
        fig2 = plt.figure(figsize=(15,15))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        ax2.set_xlabel('Principal Component'+' '+str(pc1+1), fontsize = 30)
        ax2.set_ylabel('Principal Component'+' '+str(pc2+1), fontsize = 30)
        ax2.set_zlabel('Principal Component'+' '+str(pc3+1), fontsize = 30)
        ax2.set_title('3 Component PCA', fontsize = 40)
        
        colors = ["b","g","r","c","m","y","k"]
        labels = [0,1,2,3,4,5,6]
        #label2 = [a[7:],b[7:],c[7:],d[7:],e[7:],f[7:],g[7:]]
        label2 = ['class1','class2','class3','class4','class5','class6','class7']
        
        
        for label,color,label2 in zip(labels,colors,label2):                                    
                            
            ax2.scatter(X_train_pca[np.where(label==y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))]),pc1], X_train_pca[np.where(label==y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))]),pc2], X_train_pca[np.where(label==y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))]),pc3],label= label2, c= color, s = 30)
            #ax2.scatter(X_train_pca[np.where(label==y_train),pc1], X_train_pca[np.where(label==y_train),pc2], X_train_pca[np.where(label==y_train),pc3],label= label2, c= color, s = 30)
            ax2.scatter(X_test_pca[np.where(label==y_test),pc1],X_test_pca[np.where(label==y_test),pc2],X_test_pca[np.where(label==y_test),pc3],label=label2,c=color,edgecolors='white',s=30)
            ax2.grid()
            ax2.legend(fontsize=15)
        fig2.savefig('{}/{}'.format(save_dir,"3D_Scatter"+'_{}_{}_{}'.format(pc1,pc2,pc3)))
        '''    
            
            
            
#3D SCATTER(Labeling one by one)     
        
        
        
        '''fig3 = plt.figure(figsize=(15,15))
        ax3 = fig3.add_subplot(111, projection='3d')
        
        ax3.set_xlabel('Principal Component'+' '+str(pc1+1), fontsize = 15)
        ax3.set_ylabel('Principal Component'+' '+str(pc2+1), fontsize = 15)
        ax3.set_zlabel('Principal Component'+' '+str(pc3+1), fontsize = 15)
        ax3.set_title('3 Component PCA', fontsize = 20)
        
        flag1=0
        flag2=0
        flag3=0
        flag4=0
        flag5=0
        flag6=0
        flag7=0
        
        for i in range(len(y_people)):       
        
                if y_people[i] == 0:
                    flag1 = flag1 + 1
                    label2='$'+a_list[flag1-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag1,'flag1')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='b',marker= label3,s = 6000)
                            
                elif y_people[i] == 1:
                    flag2 = flag2 + 1
                    label2='$'+b_list[flag2-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag2,'flag2')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='g',marker= label3,s = 6000)

                elif y_people[i] == 2:
                    flag3 = flag3 + 1
                    label2='$'+c_list[flag3-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag3,'flag3')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='r',marker= label3,s = 6000)
                elif y_people[i] == 3:
                    flag4 = flag4 + 1
                    label2='$'+d_list[flag4-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag4,'flag4')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='c',marker= label3,s = 6000)
                elif y_people[i] == 4:
                    flag5 = flag5 + 1
                    label2='$'+e_list[flag5-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag5,'flag5')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='m',marker= label3,s = 6000)
                elif y_people[i] == 5:
                    flag6 = flag6 + 1
                    label2='$'+f_list[flag6-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag6,'flag6')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='y',marker= label3,s = 6000)
                else:
                    flag7 = flag7 + 1
                    label2='$'+g_list[flag7-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag7,'flag7')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='k',marker= label3,s = 6000)
                print(i,'i')
        fig3.savefig('{}/{}'.format(save_dir,"3D_Scatter_Labelling"))
        samplenumber={'a':len(a_list),'b':len(b_list),'c':len(c_list),'d':len(d_list),'e':len(e_list),'f':len(f_list),'g':len(g_list)}
        data = {'pc1':X_people_pca[:,0], 'pc2':X_people_pca[:,1], 'pc3':X_people_pca[:,2], 'pc4':X_people_pca[:,3], 'pc5':X_people_pca[:,4], 'pc6':X_people_pca[:,5],'pc7':X_people_pca[:,6],'pc8':X_people_pca[:,7],'pc9':X_people_pca[:,8],'pc10':X_people_pca[:,9]}
        data2= samplenumber          
                
        list_list=[a_list,b_list,c_list,d_list,e_list,f_list,g_list]
        scoreaverage=[]        
       
        for j in range(0,10):
            averagelist=[]
            flag=0
            for i in range(0,7):
                if i == 0:
                    average=sum(X_people_pca[0:len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    flag=len(list_list[i])                    
                else:                    
                    average=sum(X_people_pca[flag:flag+len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    flag=flag+len(list_list[i])
                    
            scoreaverage.append(averagelist)
        print(scoreaverage)
        data3=scoreaverage


        data = pd.DataFrame(data)
        data2= pd.DataFrame(data2,index=[0])
        data3= pd.DataFrame(data3,columns=[str(a),str(b),str(c),str(d),str(e),str(f),str(g)])

        data.to_excel(excel_writer= save_dir+'/{}.xlsx'.format('PC',sheet_name='Principal component'))
        exfilename= save_dir+'/PC.xlsx'
        with pd.ExcelWriter(str(exfilename), mode='a', engine='openpyxl') as writer:
            data2.to_excel(writer,sheet_name='sample number')
            data3.to_excel(writer,sheet_name='score averge' )        


                
        ax3.grid()
    
    #Stack-up plot
        components=pca.components_
        y_axis=np.cumsum(np.multiply(components[stackup_pc,:],components[stackup_pc,:]))
        x_axis=range(0,144000)
        fig5,ax5=plt.subplots(figsize=(12,12))
        ax5.plot(x_axis,y_axis)
        fig5.savefig('{}/{}'.format(save_dir,"Stackup_{}".format(stackup_pc)))



        fig6,ax6=plt.subplots(figsize=(12,12))
        gradyvalue=[]
        gradxvalue=range(0,143800,200)
        for i in gradxvalue:
            grad1=(y_axis[i+200]-y_axis[i])/(x_axis[i+200]-x_axis[i])
            gradyvalue.append(grad1)
            
            
        ax6.plot(gradxvalue,gradyvalue)
        fig6.savefig('{}/{}'.format(save_dir,"Stackup_grad_{}".format(stackup_pc)))
       
    #Circle of correlation
     
        components=pca.components_        
        ev=pca.explained_variance_

        loading=components.T * np.sqrt(ev)
        an = np.linspace(0, 2 * np.pi, 100)
        fig7,ax7=plt.subplots(2,4, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})

        for i,(ax) in enumerate(ax7.ravel()):

         ax.plot(np.cos(an), np.sin(an),'r')        
         ax.scatter(loading[:,i]/np.max(loading[:,0]),loading[:,i+1]/np.max(loading[:,0]))
         ax.set_title("Circle of Correlation_PC{}_PC{}".format(i+1,i+2))
        fig7.savefig('{}/{}'.format(save_dir,"Circle of Correlation"))

    #Important Pixels Marking(PCs)


        fig8, ax8 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        for i,(component,ax) in enumerate(zip(components,ax8.ravel())):
           sortpc=np.argsort(np.square(component))[::-1]
           percent=sortpc[int(0.05*(len(sortpc))):]
           expandpc=np.expand_dims(component,axis=1)
           expandpc=np.expand_dims(expandpc,axis=2)
           stackpc=np.concatenate((expandpc,expandpc),axis=2)
           stackpc=np.concatenate((stackpc,expandpc),axis=2)
           stackpc[:,:,0]=255
           stackpc[percent,:,1]=255
           stackpc[percent,:,2]=255

           rsp_stack=np.reshape(stackpc,(360,400,3))
           ax.imshow(rsp_stack,cmap='viridis')
           ax.set_title("Important Pixels PC{}".format(i+1))
        fig8.savefig('{}/{}'.format(save_dir,"Important Pixels"))

    #Important Pixels Marking(Pixel Ranking)
    
        fig9, ax9 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        for i,(ax) in enumerate(ax9.ravel()):
           p1=components[pcpxindex]
           sortpc=np.argsort(np.square(p1))[::-1]
           percent=sortpc[int(0.05*(i+1)*(len(sortpc))):]
           expandpc=np.expand_dims(component,axis=1)
           expandpc=np.expand_dims(expandpc,axis=2)
           stackpc=np.concatenate((expandpc,expandpc),axis=2)                 
           stackpc=np.concatenate((stackpc,expandpc),axis=2)
           stackpc[:,:,0]=255
           stackpc[percent,:,1]=255
           stackpc[percent,:,2]=255
           rsp_stack=np.reshape(stackpc,(360,400,3))
           ax.imshow(rsp_stack,cmap='viridis')
           ax.set_title("Pixels for PC{}_Rank {}%".format(pcpxindex+1,(i+1)*5))
        fig9.savefig('{}/{}'.format(save_dir,"Important Pixels(Rank)_{}".format(pcpxindex+1)))'''
        return MSE_pc_list,MSE_pc_list123, MSE_pc_list4567
    

    def eigen7_test(self,a,b,c,d,e,f,g,pc1,pc2,pc3,stackup_pc,pcpxindex,pca_c,pixel):

        filename=a[7:]+b[7:]+c[7:]+d[7:]+e[7:]+f[7:]+g[7:]
        base_dir = '/home/minhyukim/PCA/'
        save_dir = base_dir+'PCA_result/'+filename

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)


        a_dir = os.path.join(base_dir,'{}'.format(a))
        b_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        e_dir = os.path.join(base_dir,'{}'.format(e))
        f_dir = os.path.join(base_dir,'{}'.format(f))
        g_dir = os.path.join(base_dir,'{}'.format(g))

        a_list = os.listdir(a_dir)
        b_list = os.listdir(b_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        e_list = os.listdir(e_dir)
        f_list = os.listdir(f_dir)
        g_list = os.listdir(g_dir)


        a_images = np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0),axis=0)
        b_images = np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        e_images = np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0),axis=0)
        f_images = np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0),axis=0)
        g_images = np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0),axis=0)
        for i in range(1,len(a_list)):
            a_images = np.concatenate((a_images,np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0),axis=0)))
        for i in range(1,len(b_list)):
            b_images = np.concatenate((b_images,np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        for i in range(1,len(e_list)):
            e_images = np.concatenate((e_images,np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0),axis=0)))
        for i in range(1,len(f_list)):
            f_images = np.concatenate((f_images,np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0),axis=0)))
        for i in range(1,len(g_list)):
            g_images = np.concatenate((g_images,np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0),axis=0)))
        images = np.concatenate((a_images,b_images,c_images,d_images,e_images,f_images,g_images),axis=0)

        a_data = np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0).flatten()),axis=0)
        b_data = np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        e_data = np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0).flatten()),axis=0)
        f_data = np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0).flatten()),axis=0)
        g_data = np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0).flatten()),axis=0)
        for i in range(1,len(a_list)):
            a_data = np.concatenate((a_data,np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(b_list)):
            b_data = np.concatenate((b_data,np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(e_list)):
            e_data = np.concatenate((e_data,np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(f_list)):
            f_data = np.concatenate((f_data,np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(g_list)):
            g_data = np.concatenate((g_data,np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((a_data,b_data,c_data,d_data,e_data,f_data,g_data),axis=0)    
        

        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(a_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(b_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target_4 = []
        for i in range(0,len(e_list)):
            target_4.append(4)
        target_5 = []
        for i in range(0,len(f_list)):
            target_5.append(5)
        target_6 = []
        for i in range(0,len(g_list)):
            target_6.append(6)
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)

        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d),'{}'.format(e), '{}'.format(f),'{}'.format(g)]

        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        '''fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()'''     

        ##


        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        X_people = data
        
        MSE_pc_list=[]           
        MSE_pc_list123=[]           
        MSE_pc_list4567=[]
        for i in range(5,pca_c+1,5) :            
            print('MSE_pc_list',MSE_pc_list)
            MSE_shuffle_list=[]
            MSE_shuffle_list123=[]
            MSE_shuffle_list4567=[]
            for j in range(3):           
                
                
                target = np.array(target)
                X_people = data
                print('X_people',len(X_people),type(X_people))
                y_people = target
                
                X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, test_size=0.2,  shuffle=True)
                
                '''
                ##향후에 k값 변경 필요시 사용
                k_list = range(1,101)
                accuracies = []
                for k in k_list:
                  classifier = KNeighborsClassifier(n_neighbors = k)
                  classifier.fit(training_data, training_labels)
                  accuracies.append(classifier.score(validation_data, validation_labels))
                plt.plot(k_list, accuracies)
                plt.xlabel("k")
                plt.ylabel("Validation Accuracy")
                plt.title("Breast Cancer Classifier Accuracy")
                plt.show()
                ''' 
                #knn = KNeighborsClassifier(n_neighbors=3)
                #knn.fit(X_train, y_train)

                pca = PCA(n_components=i+1, whiten=True, random_state=j)
                pca = pca.fit(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                #pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train)
                datavar=np.var(X_people,0)
                varmax=np.max(datavar) 
                X_train_pca = pca.transform(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                #X_train_pca = pca.transform(X_train)
                X_test_pca = pca.transform(X_test)
                print('X_test_pca',X_test_pca)
                #X_people_pca = pca.transform(X_people)

                '''  
                proba_list = []        
                accuracies = []
                classifier = KNeighborsClassifier(n_neighbors = 10)
                for i in range(80) :        
                   pca = PCA(n_components=i+2, whiten=True, random_state=0).fit(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                   X_train_pca = pca.transform(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                   X_test_pca = pca.transform(X_test)
                   classifier.fit(X_train_pca, y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                   accuracies.append(classifier.score(X_test_pca, y_test))
                   proba = classifier.predict_proba(X_test_pca)
                   proba_list.append(proba)

                fig1,ax1= plt.subplots()
                ax1.plot(range(80),accuracies) 
                ax1.set_xlabel("components")
                ax1.set_ylabel("Validation Accuracy")
                ax1.axvline(x=16, color='red', linestyle='--')
                ax1.set_title("Fluid Classifier Accuracy")
                fig1.savefig('{}/{}'.format(save_dir,"Proportion"))'''


                #fig2, ax2 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
                #for i, (component, ax2) in enumerate(zip(pca.components_, ax2.ravel())):
                #    ax2.imshow(component.reshape(image_shape), cmap='viridis')
                #    ax2.set_title("PC {}".format(i+1))

                #fig2.savefig('{}/{}'.format(save_dir,"Principal component"))


                Fluid=['R1','P1','R2','P2','R3','P3','R4','P4','R5','P5','R6','P6','R7','P7']
                xticks=[0,0.4,1,1.4,2,2.4,3,3.4,4,4.4,5,5.4,6,6.4]
                Carboref=[1,0,0,0.5,0.5 ,0,    0.33]
                CMCref=  [0,1,0,0.5,0 ,  0.5,  0.33]
                PEOref=  [0,0,1,0  ,0.5 ,0.5,  0.33]
                matric=np.vstack([Carboref,CMCref,PEOref])
                


                #k_list = range(1,10)

                classifier = KNeighborsClassifier(n_neighbors=6)
                classifier.fit(X_train_pca, y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
                #classifier.fit(X_train_pca, y_train)
                proba_list=[]
                proba = classifier.predict_proba(X_test_pca) 
                
                proba_list.append(proba)                

                #print(np.argmax(proba,axis=1))
                #print('proba',proba,'y_test',y_test)

                #f1=f1_score(y_test,np.argmax(proba,axis=1),average='macro')
                #f2=f1_score(y_test,np.argmax(proba,axis=1),average='micro')
                #f3=f1_score(y_test,np.argmax(proba,axis=1),average='weighted')


                #print('macro',f1)
                #print('micro',f2)
                #print('weighted',f3)

                Carbo_y=[[],[],[],[],[],[],[]]
                CMC_y=[[],[],[],[],[],[],[]]
                PEO_y=[[],[],[],[],[],[],[]]

                '''for i in range(len(y_test)):
                    concent=matric[:,np.where(proba[i]!=0)]*proba[i][np.where(proba[i]!=0)]
                    concent_sum=np.sum(concent,axis=-1)
                    Carbo_y[y_test[i]].append(concent_sum[0])
                    CMC_y[y_test[i]].append(concent_sum[1])
                    PEO_y[y_test[i]].append(concent_sum[2])
                    print('matric',matric[:,np.where(proba[i]!=0)],'proba',proba[i][np.where(proba[i]!=0)],'concent',concent,'y_test',y_test[i],'concent_sum',concent_sum)'''

            
                
                
                proba_list=proba_list[0]
                for i in range(len(proba_list)):
                    Carbo_y[y_test[i]].append(proba_list[i][0])
                   
                    CMC_y[y_test[i]].append(proba_list[i][1])
                    PEO_y[y_test[i]].append(proba_list[i][2])

                

                '''
                fig10,ax10 = plt.subplots()

                ax10.bar([0,1,2,3,4,5,6],Carboref,color='r',width=0.2)
                ax10.bar([0,1,2,3,4,5,6],CMCref,color='g', width=0.2,bottom=Carboref)
                ax10.bar([0,1,2,3,4,5,6],PEOref,color='b', width=0.2,bottom=[Carboref[i]+CMCref[i] for i in range(len(Carboref))])


                proba_list=np.array(proba_list)

                print(y_test)
                print('Carbo_y',Carbo_y)
                '''
                Carbo_y=[np.average(i) for i in Carbo_y]
          
                CMC_y=[np.average(i) for i in CMC_y]
                PEO_y=[np.average(i) for i in PEO_y]
                '''
                ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],Carbo_y,color='r',width=0.2,label='Carbopol')        
                ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],CMC_y,bottom=Carbo_y,color='g',width=0.2,label='CMC')
                ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],PEO_y,bottom=[Carbo_y[i]+CMC_y[i] for i in range(7)],color='b',width=0.2,label='PEO')
                ax10.set_title('Concentration ratio_{}_{}'.format(i,j))
                ax10.set_xticks(xticks)
                ax10.set_xticklabels(Fluid)
                ax10.legend()

                fig10.savefig('{}/{}'.format(save_dir,"Concentration ratio"))
                '''
                matric2=np.vstack([Carbo_y,CMC_y,PEO_y])
                MSE_list=[]
                for i in range(7):
                    #print('matric1',matric[:,i],'matric2',matric2[:,i])
                    MSE=mean_squared_error(matric[:,i],matric2[:,i])
                    MSE_list.append(MSE)
                MSE_average=np.average(MSE_list)
                #print('MSE_list',MSE_list)
                MSE_average123=np.average(MSE_list[0:3])
                MSE_average4567=np.average(MSE_list[3:-1])
                MSE_shuffle_list.append(MSE_average)
                MSE_shuffle_list123.append(MSE_average123)
                MSE_shuffle_list4567.append(MSE_average4567)
                
            MSE_pc_list.append(MSE_shuffle_list)
            MSE_pc_list123.append(MSE_shuffle_list123)
            MSE_pc_list4567.append(MSE_shuffle_list4567)
            #print('min_MSE',np.argmin(np.array(MSE_shuffle_list),axis=-1))
     
        
        fig1,ax1=plt.subplots(figsize=(12,12))
        ax1.boxplot(MSE_pc_list,[i for i in range(5,pca_c+1,5)])
        ax1.boxplot(MSE_pc_list123,[i for i in range(5,pca_c+1,5)])
        ax1.boxplot(MSE_pc_list4567,[i for i in range(5,pca_c+1,5)])
        fig2,ax2=plt.subplots(figsize=(12,12))

        for i in range(5,pca_c+1,5):            
            ax2.scatter(i*np.ones_like(MSE_pc_list123[int(i/5-1)]),MSE_pc_list123[int(i/5-1)], color='r', label='123')
        for i in range(5,pca_c+1,5):            
            ax2.scatter(i*np.ones_like(MSE_pc_list4567[int(i/5-1)]),MSE_pc_list4567[int(i/5-1)], color='g', label='4567')     
            
        ax2.set_xticks(range(5,pca_c+1,5))
        ax2.set_xlabel('PC')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.set_title("MSE for {}_frame weight".format(pixel))
        '''
#Optimal Truncation!!
        n_samples,n_features=X_train.shape
        singular_values=pca.singular_values_
        print(singular_values)
        mpmax=(1+np.sqrt(n_samples/n_features))**2
        mpmin=(1-np.sqrt(n_samples/n_features))**2
        mpmed=(mpmax+mpmin)/2
        print('mpmed',mpmed)
        threshold=4/(3**1/2)/np.sqrt(mpmed)*np.median(singular_values)
        
        print('threshold',threshold)
        components=pca.components_
       

        print('length',np.where(singular_values>threshold))
           
        
        

        variance_ratio_cumsum=np.cumsum(pca.explained_variance_ratio_)  
        
        fig13,ax13=plt.subplots()
        ax13.plot(variance_ratio_cumsum)
        ax13.set_xlabel('Number of PC')
        ax13.set_ylabel('Explained variance')
        
   
        fig14,ax14=plt.subplots()
        ax14.semilogy([i for i in range(len(singular_values))],singular_values)
        ax14.axhline(y=singular_values[-1], color='red',linestyle='--')
        ax14.text(0, singular_values[-1],"{:.2f}".format(singular_values[-1]),color='red')
        
        ax14.set_ylabel('Variance')
        
#Marcenko-Pastur 분포

        lambda_values = np.linspace(mpmin, mpmax, 500)
        alpha = n_samples/n_features
        # Marcenko-Pastur 분포 계산
        pdf_values = n_features/(2*np.pi*n_samples*alpha) * np.sqrt((mpmax - lambda_values)*(lambda_values - mpmin)) / lambda_values

        # 그래프 그리기
        fig14,ax14=plt.subplots()
        ax14.plot(lambda_values, pdf_values)
        ax14.set_title("Marcenko-Pastur Distribution (n_samples={}, n_features={}, alpha={:.2f})".format(n_samples, n_features, alpha))
        ax14.set_xlabel("Eigenvalue")
        ax14.set_ylabel("Density")
        '''
        

            

      
    
    
#3D SCATTER            
       
        
                
        '''
        fig2 = plt.figure(figsize=(15,15))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        ax2.set_xlabel('Principal Component'+' '+str(pc1+1), fontsize = 30)
        ax2.set_ylabel('Principal Component'+' '+str(pc2+1), fontsize = 30)
        ax2.set_zlabel('Principal Component'+' '+str(pc3+1), fontsize = 30)
        ax2.set_title('3 Component PCA', fontsize = 40)
        
        colors = ["b","g","r","c","m","y","k"]
        labels = [0,1,2,3,4,5,6]
        #label2 = [a[7:],b[7:],c[7:],d[7:],e[7:],f[7:],g[7:]]
        label2 = ['class1','class2','class3','class4','class5','class6','class7']
        
        
        for label,color,label2 in zip(labels,colors,label2):                                    
                            
            ax2.scatter(X_train_pca[np.where(label==y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))]),pc1], X_train_pca[np.where(label==y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))]),pc2], X_train_pca[np.where(label==y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))]),pc3],label= label2, c= color, s = 30)
            #ax2.scatter(X_train_pca[np.where(label==y_train),pc1], X_train_pca[np.where(label==y_train),pc2], X_train_pca[np.where(label==y_train),pc3],label= label2, c= color, s = 30)
            ax2.scatter(X_test_pca[np.where(label==y_test),pc1],X_test_pca[np.where(label==y_test),pc2],X_test_pca[np.where(label==y_test),pc3],label=label2,c=color,edgecolors='white',s=30)
            ax2.grid()
            ax2.legend(fontsize=15)
        fig2.savefig('{}/{}'.format(save_dir,"3D_Scatter"+'_{}_{}_{}'.format(pc1,pc2,pc3)))
        '''    
            
            
            
#3D SCATTER(Labeling one by one)     
        
        
        
        '''fig3 = plt.figure(figsize=(15,15))
        ax3 = fig3.add_subplot(111, projection='3d')
        
        ax3.set_xlabel('Principal Component'+' '+str(pc1+1), fontsize = 15)
        ax3.set_ylabel('Principal Component'+' '+str(pc2+1), fontsize = 15)
        ax3.set_zlabel('Principal Component'+' '+str(pc3+1), fontsize = 15)
        ax3.set_title('3 Component PCA', fontsize = 20)
        
        flag1=0
        flag2=0
        flag3=0
        flag4=0
        flag5=0
        flag6=0
        flag7=0
        
        for i in range(len(y_people)):       
        
                if y_people[i] == 0:
                    flag1 = flag1 + 1
                    label2='$'+a_list[flag1-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag1,'flag1')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='b',marker= label3,s = 6000)
                            
                elif y_people[i] == 1:
                    flag2 = flag2 + 1
                    label2='$'+b_list[flag2-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag2,'flag2')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='g',marker= label3,s = 6000)

                elif y_people[i] == 2:
                    flag3 = flag3 + 1
                    label2='$'+c_list[flag3-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag3,'flag3')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='r',marker= label3,s = 6000)
                elif y_people[i] == 3:
                    flag4 = flag4 + 1
                    label2='$'+d_list[flag4-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag4,'flag4')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='c',marker= label3,s = 6000)
                elif y_people[i] == 4:
                    flag5 = flag5 + 1
                    label2='$'+e_list[flag5-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag5,'flag5')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='m',marker= label3,s = 6000)
                elif y_people[i] == 5:
                    flag6 = flag6 + 1
                    label2='$'+f_list[flag6-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag6,'flag6')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='y',marker= label3,s = 6000)
                else:
                    flag7 = flag7 + 1
                    label2='$'+g_list[flag7-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag7,'flag7')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='k',marker= label3,s = 6000)
                print(i,'i')
        fig3.savefig('{}/{}'.format(save_dir,"3D_Scatter_Labelling"))
        samplenumber={'a':len(a_list),'b':len(b_list),'c':len(c_list),'d':len(d_list),'e':len(e_list),'f':len(f_list),'g':len(g_list)}
        data = {'pc1':X_people_pca[:,0], 'pc2':X_people_pca[:,1], 'pc3':X_people_pca[:,2], 'pc4':X_people_pca[:,3], 'pc5':X_people_pca[:,4], 'pc6':X_people_pca[:,5],'pc7':X_people_pca[:,6],'pc8':X_people_pca[:,7],'pc9':X_people_pca[:,8],'pc10':X_people_pca[:,9]}
        data2= samplenumber          
                
        list_list=[a_list,b_list,c_list,d_list,e_list,f_list,g_list]
        scoreaverage=[]        
       
        for j in range(0,10):
            averagelist=[]
            flag=0
            for i in range(0,7):
                if i == 0:
                    average=sum(X_people_pca[0:len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    flag=len(list_list[i])                    
                else:                    
                    average=sum(X_people_pca[flag:flag+len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    flag=flag+len(list_list[i])
                    
            scoreaverage.append(averagelist)
        print(scoreaverage)
        data3=scoreaverage


        data = pd.DataFrame(data)
        data2= pd.DataFrame(data2,index=[0])
        data3= pd.DataFrame(data3,columns=[str(a),str(b),str(c),str(d),str(e),str(f),str(g)])

        data.to_excel(excel_writer= save_dir+'/{}.xlsx'.format('PC',sheet_name='Principal component'))
        exfilename= save_dir+'/PC.xlsx'
        with pd.ExcelWriter(str(exfilename), mode='a', engine='openpyxl') as writer:
            data2.to_excel(writer,sheet_name='sample number')
            data3.to_excel(writer,sheet_name='score averge' )        


                
        ax3.grid()
    
    #Stack-up plot
        components=pca.components_
        y_axis=np.cumsum(np.multiply(components[stackup_pc,:],components[stackup_pc,:]))
        x_axis=range(0,144000)
        fig5,ax5=plt.subplots(figsize=(12,12))
        ax5.plot(x_axis,y_axis)
        fig5.savefig('{}/{}'.format(save_dir,"Stackup_{}".format(stackup_pc)))



        fig6,ax6=plt.subplots(figsize=(12,12))
        gradyvalue=[]
        gradxvalue=range(0,143800,200)
        for i in gradxvalue:
            grad1=(y_axis[i+200]-y_axis[i])/(x_axis[i+200]-x_axis[i])
            gradyvalue.append(grad1)
            
            
        ax6.plot(gradxvalue,gradyvalue)
        fig6.savefig('{}/{}'.format(save_dir,"Stackup_grad_{}".format(stackup_pc)))
       
    #Circle of correlation
     
        components=pca.components_        
        ev=pca.explained_variance_

        loading=components.T * np.sqrt(ev)
        an = np.linspace(0, 2 * np.pi, 100)
        fig7,ax7=plt.subplots(2,4, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})

        for i,(ax) in enumerate(ax7.ravel()):

         ax.plot(np.cos(an), np.sin(an),'r')        
         ax.scatter(loading[:,i]/np.max(loading[:,0]),loading[:,i+1]/np.max(loading[:,0]))
         ax.set_title("Circle of Correlation_PC{}_PC{}".format(i+1,i+2))
        fig7.savefig('{}/{}'.format(save_dir,"Circle of Correlation"))

    #Important Pixels Marking(PCs)


        fig8, ax8 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        for i,(component,ax) in enumerate(zip(components,ax8.ravel())):
           sortpc=np.argsort(np.square(component))[::-1]
           percent=sortpc[int(0.05*(len(sortpc))):]
           expandpc=np.expand_dims(component,axis=1)
           expandpc=np.expand_dims(expandpc,axis=2)
           stackpc=np.concatenate((expandpc,expandpc),axis=2)
           stackpc=np.concatenate((stackpc,expandpc),axis=2)
           stackpc[:,:,0]=255
           stackpc[percent,:,1]=255
           stackpc[percent,:,2]=255

           rsp_stack=np.reshape(stackpc,(360,400,3))
           ax.imshow(rsp_stack,cmap='viridis')
           ax.set_title("Important Pixels PC{}".format(i+1))
        fig8.savefig('{}/{}'.format(save_dir,"Important Pixels"))

    #Important Pixels Marking(Pixel Ranking)
    
        fig9, ax9 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        for i,(ax) in enumerate(ax9.ravel()):
           p1=components[pcpxindex]
           sortpc=np.argsort(np.square(p1))[::-1]
           percent=sortpc[int(0.05*(i+1)*(len(sortpc))):]
           expandpc=np.expand_dims(component,axis=1)
           expandpc=np.expand_dims(expandpc,axis=2)
           stackpc=np.concatenate((expandpc,expandpc),axis=2)                 
           stackpc=np.concatenate((stackpc,expandpc),axis=2)
           stackpc[:,:,0]=255
           stackpc[percent,:,1]=255
           stackpc[percent,:,2]=255
           rsp_stack=np.reshape(stackpc,(360,400,3))
           ax.imshow(rsp_stack,cmap='viridis')
           ax.set_title("Pixels for PC{}_Rank {}%".format(pcpxindex+1,(i+1)*5))
        fig9.savefig('{}/{}'.format(save_dir,"Important Pixels(Rank)_{}".format(pcpxindex+1)))'''
        return MSE_pc_list,MSE_pc_list123, MSE_pc_list4567, target, data  
    
    
    def accuracy_Plot(self,a,b,c,d,e,f,g,pca_c):
        y_people=self.component(a,b,c,d,e,f,g,pca_c)[7]
        X_people=self.component(a,b,c,d,e,f,g,pca_c)[5]
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, test_size=0.2,  shuffle=True)     
        
   

      
        fig1,ax1=plt.subplots(2,5,figsize=(38,15),dpi=300)
                  
        k_list = range(1,20)
        y_ticks=[0,0.2,0.4,0.6,0.8,1.0]
        x_ticks=[0,5,10,15,20]  


        for i,ax in zip(range(0,pca_c+1,5),ax1.ravel()):
            pca = PCA(n_components=i+1, whiten=True, random_state=0).fit(X_train)
            X_train_pca=pca.transform(X_train)
            X_test_pca=pca.transform(X_test)            
            accuracies = []
            for k in k_list:
                classifier = KNeighborsClassifier(n_neighbors = k)
                classifier.fit(X_train_pca, y_train)
                accuracies.append(classifier.score(X_test_pca, y_test))
            ax.plot(k_list, accuracies, color=(0.7,0.1,0.1,0.7))
            ax.set_xlabel("k",fontsize=20)
            ax.set_xticks(x_ticks,x_ticks,fontsize=20)
            ax.set_yticks(y_ticks,y_ticks,fontsize=20)
            ax.set_ylabel("Validation Accuracy",fontsize=20)
            ax.set_title("Principal component {}".format(i+1),fontsize=20)
        fig1.suptitle("Classification accuracy acc.to PC and K",fontsize=30)


        
        
   
    
    
    
    def ratio_Plot(self,a,b,c,d,e,f,g,pc1,pc2,pc3,stackup_pc,pcpxindex,pca_c,pixel,random_state):    
        
        '''
        
    
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        c : str
            분류유체3의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
    
        Returns
        -------
        None.
    
        '''
        filename=a[7:]+b[7:]+c[7:]+d[7:]+e[7:]+f[7:]+g[7:]
        base_dir = '/home/minhyukim/PCA/'
        save_dir = base_dir+'PCA_result/'+filename

        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)


        a_dir = os.path.join(base_dir,'{}'.format(a))
        b_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
        e_dir = os.path.join(base_dir,'{}'.format(e))
        f_dir = os.path.join(base_dir,'{}'.format(f))
        g_dir = os.path.join(base_dir,'{}'.format(g))
        
        a_list = os.listdir(a_dir)
        b_list = os.listdir(b_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
        e_list = os.listdir(e_dir)
        f_list = os.listdir(f_dir)
        g_list = os.listdir(g_dir)
   
        
        a_images = np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0),axis=0)
        b_images = np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
        e_images = np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0),axis=0)
        f_images = np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0),axis=0)
        g_images = np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0),axis=0)
        for i in range(1,len(a_list)):
            a_images = np.concatenate((a_images,np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0),axis=0)))
        for i in range(1,len(b_list)):
            b_images = np.concatenate((b_images,np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        for i in range(1,len(e_list)):
            e_images = np.concatenate((e_images,np.expand_dims(cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0),axis=0)))
        for i in range(1,len(f_list)):
            f_images = np.concatenate((f_images,np.expand_dims(cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0),axis=0)))
        for i in range(1,len(g_list)):
            g_images = np.concatenate((g_images,np.expand_dims(cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0),axis=0)))
        images = np.concatenate((a_images,b_images,c_images,d_images,e_images,f_images,g_images),axis=0)
        
        a_data = np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0).flatten()),axis=0)
        b_data = np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
        e_data = np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[0])),0).flatten()),axis=0)
        f_data = np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[0])),0).flatten()),axis=0)
        g_data = np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[0])),0).flatten()),axis=0)
        for i in range(1,len(a_list)):
            a_data = np.concatenate((a_data,np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(b_list)):
            b_data = np.concatenate((b_data,np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(e_list)):
            e_data = np.concatenate((e_data,np.expand_dims((cv2.imread(os.path.join(e_dir,'{}'.format(e_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(f_list)):
            f_data = np.concatenate((f_data,np.expand_dims((cv2.imread(os.path.join(f_dir,'{}'.format(f_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(g_list)):
            g_data = np.concatenate((g_data,np.expand_dims((cv2.imread(os.path.join(g_dir,'{}'.format(g_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((a_data,b_data,c_data,d_data,e_data,f_data,g_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(a_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(b_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
        target_4 = []
        for i in range(0,len(e_list)):
            target_4.append(4)
        target_5 = []
        for i in range(0,len(f_list)):
            target_5.append(5)
        target_6 = []
        for i in range(0,len(g_list)):
            target_6.append(6)
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d),'{}'.format(e), '{}'.format(f),'{}'.format(g)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 + target_4 + target_5 + target_6##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        X_people = data
        y_people = target
        
       

      
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people,stratify=y_people, test_size=0.1, random_state=random_state)


        pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
        #pca = PCA(n_components=pca_c, whiten=True, random_state=0).fit(X_train)
        datavar=np.var(X_people,0)
        varmax=np.max(datavar) 
        X_train_pca = pca.transform(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
        #X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_people_pca = pca.transform(X_people)

        '''  
        proba_list = []        
        accuracies = []
        classifier = KNeighborsClassifier(n_neighbors = 10)
        for i in range(80) :        
           pca = PCA(n_components=i+2, whiten=True, random_state=0).fit(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
           X_train_pca = pca.transform(X_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
           X_test_pca = pca.transform(X_test)
           classifier.fit(X_train_pca, y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
           accuracies.append(classifier.score(X_test_pca, y_test))
           proba = classifier.predict_proba(X_test_pca)
           proba_list.append(proba)

        fig1,ax1= plt.subplots()
        ax1.plot(range(80),accuracies) 
        ax1.set_xlabel("components")
        ax1.set_ylabel("Validation Accuracy")
        ax1.axvline(x=16, color='red', linestyle='--')
        ax1.set_title("Fluid Classifier Accuracy")
        fig1.savefig('{}/{}'.format(save_dir,"Proportion"))'''


        #fig2, ax2 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        #for i, (component, ax2) in enumerate(zip(pca.components_, ax2.ravel())):
        #    ax2.imshow(component.reshape(image_shape), cmap='viridis')
        #    ax2.set_title("PC {}".format(i+1))

        #fig2.savefig('{}/{}'.format(save_dir,"Principal component"))


        Fluid=['L1','E1','L2','E2','L3','E3','L4','E4','L5','E5','L6','E6','L7','E7']
        xticks=[0,0.4,1,1.4,2,2.4,3,3.4,4,4.4,5,5.4,6,6.4]
        yticks=[0,0.2,0.4,0.6,0.8,1.0]
        Carboref=[1,0,0,0.5,0.5 ,0,    1/3]
        CMCref=  [0,1,0,0.5,0 ,  0.5,  1/3]
        PEOref=  [0,0,1,0  ,0.5 ,0.5,  1/3]
        matric=np.vstack([Carboref,CMCref,PEOref])
        #print(matric)


            #k_list = range(1,10)

        classifier = KNeighborsClassifier(n_neighbors=6)
        classifier.fit(X_train_pca, y_train[np.where((y_train==0)|(y_train==1)|(y_train==2))])
        #classifier.fit(X_train_pca, y_train)
        proba = classifier.predict_proba(X_test_pca) 


        proba_list=[]

        #print(np.argmax(proba,axis=1))
        #print('proba',proba,'y_test',y_test)

        #f1=f1_score(y_test,np.argmax(proba,axis=1),average='macro')
        #f2=f1_score(y_test,np.argmax(proba,axis=1),average='micro')
        #f3=f1_score(y_test,np.argmax(proba,axis=1),average='weighted')


        #print('macro',f1)
        #print('micro',f2)
        #print('weighted',f3)

        Carbo_y=[[],[],[],[],[],[],[]]
        CMC_y=[[],[],[],[],[],[],[]]
        PEO_y=[[],[],[],[],[],[],[]]

        '''for i in range(len(y_test)):
            concent=matric[:,np.where(proba[i]!=0)]*proba[i][np.where(proba[i]!=0)]
            concent_sum=np.sum(concent,axis=-1)
            Carbo_y[y_test[i]].append(concent_sum[0])

            CMC_y[y_test[i]].append(concent_sum[1])
            PEO_y[y_test[i]].append(concent_sum[2])
            print('matric',matric[:,np.where(proba[i]!=0)],'proba',proba[i][np.where(proba[i]!=0)],'concent',concent,'y_test',y_test[i],'concent_sum',concent_sum)'''

        proba_list.append(proba)
        #print(proba_list)

        proba_list=proba_list[0]
        for i in range(len(proba_list)):
            Carbo_y[y_test[i]].append(proba_list[i][0])
            CMC_y[y_test[i]].append(proba_list[i][1])
            PEO_y[y_test[i]].append(proba_list[i][2])




        fig10,ax10 = plt.subplots()

        ax10.bar([0,1,2,3,4,5,6],Carboref,color='r',width=0.2)
        ax10.bar([0,1,2,3,4,5,6],CMCref,color='g', width=0.2,bottom=Carboref)
        ax10.bar([0,1,2,3,4,5,6],PEOref,color='b', width=0.2,bottom=[Carboref[i]+CMCref[i] for i in range(len(Carboref))])


        proba_list=np.array(proba_list)

        Carbo_y=[np.average(i) for i in Carbo_y]        
        CMC_y=[np.average(i) for i in CMC_y]
        PEO_y=[np.average(i) for i in PEO_y]

      
            

        ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],Carbo_y,color='r',width=0.2,label='Carbopol')        
        ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],CMC_y,bottom=Carbo_y,color='g',width=0.2,label='CMC')
        ax10.bar([0.4,1.4,2.4,3.4,4.4,5.4,6.4],PEO_y,bottom=[Carbo_y[i]+CMC_y[i] for i in range(7)],color='b',width=0.2,label='PEO')
      
        ax10.set_xticks(xticks,Fluid,fontsize=15)
        ax10.set_yticks(yticks,yticks,fontsize=15)
        ax10.set_ylabel('Ratio',fontsize=15)
        #ax10.set_xticklabels(Fluid)
        ax10.legend(fontsize=10,loc='upper right')
        
    def box_Plot0(self,pca_c):
        
        pixel1=self.eigen7('220907_Carbopol_noweight','220907_CMC_noweight','220907_PEO_noweight','220907_CMC1,Carbopol1_noweight','220907_PEO1,Carbopol1_noweight','220907_PEO1,CMC1_noweight','220907_PEO1,CMC1,Carbopol1_noweight',0,1,2,1,1,pca_c,0)
        pixel2=self.eigen7('230413_Carbopol_100weight','230411_CMC_100weight','230411_PEO_100weight','230412_CMC1_Carbopol1_100weight','230412_PEO1,Carbopol1_100weight','230412_PEO1,CMC1_100weight','230412_PEO1,CMC1,Carbopol1_100weight',0,1,2,1,1,pca_c,10)
        pixel3=self.eigen7('230508_Carbopol_100weight_30frame','230509_CMC_100_weight_30frame','230509_PEO_100weight_30frame','230509_CMC1_Carbopol1_100weight_30frame','230508_PEO1_Carbopol1_100weight_30frame','230509_PEO1_CMC1_100weight_30frame','230509_PEO1_CMC1_Carbopol1_100weight_30frame',0,1,2,1,1,pca_c,30)
        pixel4=self.eigen7('230508_Carbopol_100weight_50frame','230509_CMC_100_weight_50frame','230509_PEO_100weight_50frame','230509_CMC1_Carbopol1_100weight_50frame','230508_PEO1_Carbopol1_100weight_50frame','230509_PEO1_CMC1_100weight_50frame','230509_PEO1_CMC1_Carbopol1_100weight_50frame',0,1,2,1,1,pca_c,50)

        '''pixel1=self.eigen7('220907_Carbopol_noweight_1','220907_CMC_noweight_1','220907_PEO_noweight_1','220907_CMC1,Carbopol1_noweight_1','220907_PEO1,Carbopol1_noweight_1','220907_PEO1,CMC1_noweight_1','220907_PEO1,CMC1,Carbopol1_noweight_1',0,1,2,1,1,pca_c,0)
        pixel2=self.eigen7('230413_Carbopol_100weight_1','230411_CMC_100weight_1','230411_PEO_100weight_1','230412_CMC1_Carbopol1_100weight_1','230412_PEO1,Carbopol1_100weight_1','230412_PEO1,CMC1_100weight_1','230412_PEO1,CMC1,Carbopol1_100weight_1',0,1,2,1,1,pca_c,10)
        pixel3=self.eigen7('230508_Carbopol_100weight_30frame_1','230509_CMC_100_weight_30frame_1','230509_PEO_100weight_30frame_1','230509_CMC1_Carbopol1_100weight_30frame_1','230508_PEO1_Carbopol1_100weight_30frame_1','230509_PEO1_CMC1_100weight_30frame_1','230509_PEO1_CMC1_Carbopol1_100weight_30frame_1',0,1,2,1,1,pca_c,30)
        pixel4=self.eigen7('230508_Carbopol_100weight_50frame_1','230509_CMC_100_weight_50frame_1','230509_PEO_100weight_50frame_1','230509_CMC1_Carbopol1_100weight_50frame_1','230508_PEO1_Carbopol1_100weight_50frame_1','230509_PEO1_CMC1_100weight_50frame_1','230509_PEO1_CMC1_Carbopol1_100weight_50frame_1',0,1,2,1,1,pca_c,50)'''
              
        
        
        
        pixels=[pixel1,pixel2,pixel3,pixel4]
        
        colors1=['#BF4F5E','#92B68C','#B5D2D9','#C7A497']
        colors2=['#F56578','#B8E6B1','#CCEDF5','#EDC3B4']
        labels1=['0F 100W','10F 100W','30F 100W','50F 100W']
        labels2=['0_123','10_123','30_123','50_123','0_4567','10_4567','30_4567','50_4567']
        loca0=[-2,-1,1,2]
        loca1=[-3,-1,1,3]
        loca2=[-2,0,2,4]
        boxs=[]
        fig1,ax1=plt.subplots(figsize=(20,12),dpi=300)
        for pixel,color,label,loc in zip(pixels,colors1,labels2,loca1):
            flierprops = dict(marker='o', markerfacecolor=color, markersize=4)                
                
            box=ax1.boxplot(pixel[0],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=1)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])
            
        for pixel,color,label,loc in zip(pixels,colors1,labels1,loca0):
            flierprops = dict(marker='x', markerfacecolor=color, markersize=4)                
                
            #box=ax1.boxplot(pixel[2],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=0.5)
            
            #for i in box['boxes']:
            #    i.set_facecolor(color)
            #    i.set(edgecolor='w', linestyle='-', linewidth=1)
            #boxs.append(box['boxes'])
            
            
        '''for pixel,color,label,loc in zip(pixels,colors2,labels1,loca2):
            flierprops = dict(marker='x', markerfacecolor=color, markersize=4)                
                
            box=ax1.boxplot(pixel[2],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=0.5)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])'''
            
        ax1.legend([boxs[0][0],boxs[1][0],boxs[2][0],boxs[3][0]],labels1,loc='upper right',fontsize=15)
                    
        #ax1.legend([boxs[0][0],boxs[1][0],boxs[2][0],boxs[3][0],boxs[4][0],boxs[5][0],boxs[6][0],boxs[7][0]],labels2,loc='upper right',fontsize=15)
        
        ax1.set_xticks([i for i in range(5,pca_c+1,10)],[i for i in range(5,pca_c+1,10)],fontsize=15)        
        ax1.set_xlabel('PC',fontsize=15)
        ax1.set_ylabel('MSE',fontsize=15)
        ax1.set_ylim(-0.03,0.30) 
        ax1.tick_params(axis='y', labelsize=15)
        ax1.set_xlabel('Weight',fontsize=15)
        ax1.set_ylabel('MSE',fontsize=15)
        
        
        
        
        
    def box_Plot1(self,pca_c):
        
        '''pixel1=self.eigen7('220907_Carbopol_noweight','220907_CMC_noweight','220907_PEO_noweight','220907_CMC1,Carbopol1_noweight','220907_PEO1,Carbopol1_noweight','220907_PEO1,CMC1_noweight','220907_PEO1,CMC1,Carbopol1_noweight',0,1,2,1,1,pca_c,0)
        pixel2=self.eigen7('230413_Carbopol_100weight','230411_CMC_100weight','230411_PEO_100weight','230412_CMC1_Carbopol1_100weight','230412_PEO1,Carbopol1_100weight','230412_PEO1,CMC1_100weight','230412_PEO1,CMC1,Carbopol1_100weight',0,1,2,1,1,pca_c,10)
        pixel3=self.eigen7('230508_Carbopol_100weight_30frame','230509_CMC_100_weight_30frame','230509_PEO_100weight_30frame','230509_CMC1_Carbopol1_100weight_30frame','230508_PEO1_Carbopol1_100weight_30frame','230509_PEO1_CMC1_100weight_30frame','230509_PEO1_CMC1_Carbopol1_100weight_30frame',0,1,2,1,1,pca_c,30)
        pixel4=self.eigen7('230508_Carbopol_100weight_50frame','230509_CMC_100_weight_50frame','230509_PEO_100weight_50frame','230509_CMC1_Carbopol1_100weight_50frame','230508_PEO1_Carbopol1_100weight_50frame','230509_PEO1_CMC1_100weight_50frame','230509_PEO1_CMC1_Carbopol1_100weight_50frame',0,1,2,1,1,pca_c,50)'''

        pixel1=self.eigen7('220907_Carbopol_noweight_1','220907_CMC_noweight_1','220907_PEO_noweight_1','220907_CMC1,Carbopol1_noweight_1','220907_PEO1,Carbopol1_noweight_1','220907_PEO1,CMC1_noweight_1','220907_PEO1,CMC1,Carbopol1_noweight_1',0,1,2,1,1,pca_c,0)
        pixel2=self.eigen7('230413_Carbopol_100weight_1','230411_CMC_100weight_1','230411_PEO_100weight_1','230412_CMC1_Carbopol1_100weight_1','230412_PEO1,Carbopol1_100weight_1','230412_PEO1,CMC1_100weight_1','230412_PEO1,CMC1,Carbopol1_100weight_1',0,1,2,1,1,pca_c,10)
        pixel3=self.eigen7('230508_Carbopol_100weight_30frame_1','230509_CMC_100_weight_30frame_1','230509_PEO_100weight_30frame_1','230509_CMC1_Carbopol1_100weight_30frame_1','230508_PEO1_Carbopol1_100weight_30frame_1','230509_PEO1_CMC1_100weight_30frame_1','230509_PEO1_CMC1_Carbopol1_100weight_30frame_1',0,1,2,1,1,pca_c,30)
        pixel4=self.eigen7('230508_Carbopol_100weight_50frame_1','230509_CMC_100_weight_50frame_1','230509_PEO_100weight_50frame_1','230509_CMC1_Carbopol1_100weight_50frame_1','230508_PEO1_Carbopol1_100weight_50frame_1','230509_PEO1_CMC1_100weight_50frame_1','230509_PEO1_CMC1_Carbopol1_100weight_50frame_1',0,1,2,1,1,pca_c,50)
              
        
        
        
        pixels=[pixel1,pixel2,pixel3,pixel4]
        
        colors1=['#BF4F5E','#92B68C','#B5D2D9','#C7A497']
        colors2=['#F56578','#B8E6B1','#CCEDF5','#EDC3B4']
        labels1=['0F 100W','10F 100W','30F 100W','50F 100W']
        labels2=['0_123','10_123','30_123','50_123','0_4567','10_4567','30_4567','50_4567']
        loca0=[-2,-1,1,2]
        loca1=[-3,-1,1,3]
        loca2=[-2,0,2,4]
        boxs=[]
        fig1,ax1=plt.subplots(figsize=(20,12),dpi=300)
        for pixel,color,label,loc in zip(pixels,colors1,labels2,loca1):
            flierprops = dict(marker='o', markerfacecolor=color, markersize=4)                
                
            box=ax1.boxplot(pixel[0],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=1)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])
            
        for pixel,color,label,loc in zip(pixels,colors1,labels1,loca0):
            flierprops = dict(marker='x', markerfacecolor=color, markersize=4)                
                
            #box=ax1.boxplot(pixel[2],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=0.5)
            
            #for i in box['boxes']:
            #    i.set_facecolor(color)
            #    i.set(edgecolor='w', linestyle='-', linewidth=1)
            #boxs.append(box['boxes'])
            
            
        '''for pixel,color,label,loc in zip(pixels,colors2,labels1,loca2):
            flierprops = dict(marker='x', markerfacecolor=color, markersize=4)                
                
            box=ax1.boxplot(pixel[2],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=0.5)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])'''
            
        ax1.legend([boxs[0][0],boxs[1][0],boxs[2][0],boxs[3][0]],labels1,loc='upper right',fontsize=15)
                    
        #ax1.legend([boxs[0][0],boxs[1][0],boxs[2][0],boxs[3][0],boxs[4][0],boxs[5][0],boxs[6][0],boxs[7][0]],labels2,loc='upper right',fontsize=15)
        
        ax1.set_xticks([i for i in range(5,pca_c+1,10)],[i for i in range(5,pca_c+1,10)],fontsize=15)        
        ax1.set_xlabel('PC',fontsize=15)
        ax1.set_ylabel('MSE',fontsize=15)
        ax1.set_ylim(-0.03,0.30) 
        ax1.tick_params(axis='y', labelsize=15)
        ax1.set_xlabel('Weight',fontsize=15)
        ax1.set_ylabel('MSE',fontsize=15)        

        
        

        
    def box_Plot2(self,pca_c):
        
        pixel1=self.eigen7('220907_Carbopol_noweight','220907_CMC_noweight','220907_PEO_noweight','220907_CMC1,Carbopol1_noweight','220907_PEO1,Carbopol1_noweight','220907_PEO1,CMC1_noweight','220907_PEO1,CMC1,Carbopol1_noweight',0,1,2,1,1,pca_c,0)
        pixel2=self.eigen7('230620_Carbopol_10frame_50weight','230620_CMC_10frame_50weight','230620_PEO_10frame_50weight','230620_Carbopol1_CMC1_10frame_50weight','230620_Carbopol1_PEO1_10frame_50weight','230620_PEO1_CMC1_10frame_50weight','230620_Carbopol1_CMC1_PEO1_10frame_50weight',0,1,2,1,1,pca_c,0)
        pixel3=self.eigen7('230413_Carbopol_100weight','230411_CMC_100weight','230411_PEO_100weight','230412_CMC1_Carbopol1_100weight','230412_PEO1,Carbopol1_100weight','230412_PEO1,CMC1_100weight','230412_PEO1,CMC1,Carbopol1_100weight',0,1,2,1,1,pca_c,0)
        pixel4=self.eigen7('230621_Carbopol_10frame_150weight','230621_CMC_10frame_150weight','230621_PEO_10frame_150weight','230621_Carbopol1_CMC1_10frame_150weight','230621_Carbopol1_PEO1_10frame_150weight','230621_PEO1_CMC1_10frame_150weight','230621_Carbopol1_CMC1_PEO1_10frame_150weight',0,1,2,1,1,pca_c,0)
        
        pixel5=self.eigen7('230622_Carbopol_10frame_200weight','230622_CMC_10frame_200weight','230622_PEO_10frame_200weight','230622_Carbopol1_CMC1_10frame_200weight','230622_Carbopol1_PEO1_10frame_200weight','230622_PEO1_CMC1_10frame_200weight','230622_Carbopol1_CMC1_PEO1_10frame_200weight',0,1,2,1,1,pca_c,0)
        #pixel3=self.eigen7('230508_Carbopol_100weight_30frame','230509_CMC_100_weight_30frame','230509_PEO_100weight_30frame','230509_CMC1_Carbopol1_100weight_30frame','230508_PEO1_Carbopol1_100weight_30frame','230509_PEO1_CMC1_100weight_30frame','230509_PEO1_CMC1_Carbopol1_100weight_30frame',0,1,2,1,1,pca_c,30)
        #pixel4=self.eigen7('230508_Carbopol_100weight_50frame','230509_CMC_100_weight_50frame','230509_PEO_100weight_50frame','230509_CMC1_Carbopol1_100weight_50frame','230508_PEO1_Carbopol1_100weight_50frame','230509_PEO1_CMC1_100weight_50frame','230509_PEO1_CMC1_Carbopol1_100weight_50frame',0,1,2,1,1,pca_c,50)

        '''pixel1=self.eigen7('220907_Carbopol_noweight_1','220907_CMC_noweight_1','220907_PEO_noweight_1','220907_CMC1,Carbopol1_noweight_1','220907_PEO1,Carbopol1_noweight_1','220907_PEO1,CMC1_noweight_1','220907_PEO1,CMC1,Carbopol1_noweight_1',0,1,2,1,1,pca_c,0)
        pixel2=self.eigen7('230413_Carbopol_100weight_1','230411_CMC_100weight_1','230411_PEO_100weight_1','230412_CMC1_Carbopol1_100weight_1','230412_PEO1,Carbopol1_100weight_1','230412_PEO1,CMC1_100weight_1','230412_PEO1,CMC1,Carbopol1_100weight_1',0,1,2,1,1,pca_c,10)
        pixel3=self.eigen7('230508_Carbopol_100weight_30frame_1','230509_CMC_100_weight_30frame_1','230509_PEO_100weight_30frame_1','230509_CMC1_Carbopol1_100weight_30frame_1','230508_PEO1_Carbopol1_100weight_30frame_1','230509_PEO1_CMC1_100weight_30frame_1','230509_PEO1_CMC1_Carbopol1_100weight_30frame_1',0,1,2,1,1,pca_c,30)
        pixel4=self.eigen7('230508_Carbopol_100weight_50frame_1','230509_CMC_100_weight_50frame_1','230509_PEO_100weight_50frame_1','230509_CMC1_Carbopol1_100weight_50frame_1','230508_PEO1_Carbopol1_100weight_50frame_1','230509_PEO1_CMC1_100weight_50frame_1','230509_PEO1_CMC1_Carbopol1_100weight_50frame_1',0,1,2,1,1,pca_c,50)'''
              
        
        
        
        pixels=[pixel1,pixel2,pixel3,pixel4,pixel5]
        colors0=['#BF4F5E','#92B68C','#B5D2D9','#C7A497','#3B75AD']

        labels1=['10F 0W','10F 50W','10F 100W','10F 150W','10F 200W']
      
        loca0=[0,1,2,3,4]
        boxs=[]
        fig1,ax1=plt.subplots(figsize=(20,12),dpi=300)
        for pixel,color,label,loc in zip(pixels,colors0,labels1,loca0):
            flierprops = dict(marker='o', markerfacecolor=color, markersize=4)      
            box=ax1.boxplot(pixel[0][2],positions=[loc],patch_artist=True,showfliers=True, flierprops=flierprops,widths=1)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])
            
        '''for pixel,color,label,loc in zip(pixels,colors2,labels1,loca2):
            flierprops = dict(marker='x', markerfacecolor=color, markersize=4)                
                
            box=ax1.boxplot(pixel[2],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=0.5)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])'''
            
            
        '''for pixel,color,label,loc in zip(pixels,colors2,labels1,loca2):
            flierprops = dict(marker='x', markerfacecolor=color, markersize=4)                
                
            box=ax1.boxplot(pixel[2],positions=[i for i in range(5+loc,pca_c+1+loc,10)],patch_artist=True,showfliers=True, flierprops=flierprops,widths=0.5)
            
            for i in box['boxes']:
                i.set_facecolor(color)
                i.set(edgecolor='w', linestyle='-', linewidth=1)
            boxs.append(box['boxes'])'''
            
        #ax1.legend([boxs[0][0],boxs[1][0],boxs[2][0],boxs[3][0]],labels1,loc='upper right',fontsize=15)
        print('boxs',boxs)
        ax1.legend([boxs[0][0],boxs[1][0],boxs[2][0],boxs[3][0],boxs[4][0]],labels1,loc='upper right',fontsize=15)
        
        ax1.set_xticks(loca0)
        ax1.set_xticklabels(['0','50','100','150','200'],fontsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        ax1.set_xlabel('Weight',fontsize=15)
        ax1.set_ylabel('MSE',fontsize=15)
      
                
        
            
    def scatter_Plot(self,pca_c):
        
        pixel1=self.eigen7('220907_Carbopol_noweight','220907_CMC_noweight','220907_PEO_noweight','220907_CMC1,Carbopol1_noweight','220907_PEO1,Carbopol1_noweight','220907_PEO1,CMC1_noweight','220907_PEO1,CMC1,Carbopol1_noweight',0,1,2,1,1,pca_c) 
        pixel2=self.eigen7('230413_Carbopol_100weight','230411_CMC_100weight','230411_PEO_100weight','230412_CMC1_Carbopol1_100weight','230412_PEO1,Carbopol1_100weight','230412_PEO1,CMC1_100weight','230412_PEO1,CMC1,Carbopol1_100weight',0,1,2,1,1,pca_c)
        pixel3=self.eigen7('230508_Carbopol_100weight_30frame','230509_CMC_100_weight_30frame','230509_PEO_100weight_30frame','230509_CMC1_Carbopol1_100weight_30frame','230508_PEO1_Carbopol1_100weight_30frame','230509_PEO1_CMC1_100weight_30frame','230509_PEO1_CMC1_Carbopol1_100weight_30frame',0,1,2,1,1,pca_c)
        pixel4=self.eigen7('230508_Carbopol_100weight_50frame','230509_CMC_100_weight_50frame','230509_PEO_100weight_50frame','230509_CMC1_Carbopol1_100weight_50frame','230508_PEO1_Carbopol1_100weight_50frame','230509_PEO1_CMC1_100weight_50frame','230509_PEO1_CMC1_Carbopol1_100weight_50frame',0,1,2,1,1,pca_c)

                               
        
        
        
        pixels=[pixel1,pixel2,pixel3,pixel4]
        colors=['r','g','b','y']
        labels=['no','10','30','50']
        pcs=range(9,pca_c,10)
        boxs=[]
        fig1,ax1=plt.subplots(figsize=(12,12))
        for pixel,color,label in zip(pixels,colors,labels):
            for pc in pcs:
                ax1.scatter(pc*np.ones_like(pixel)+(pc+1)/10-1,pixel,label= label, c= color, s = 30)                       
        ax1.set_xticks(pcs)
  
        
        ax1.legend(loc='upper right')
            
            
        
    def augmentation(self,a,b,c,d,e,f,g,pc):
        save_dir='PCA/Data_augmentation/{}_{}'.format(a[-7:],pc)
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)
        comp=self.component(a,b,c,d,e,f,g,pc)        
        components=comp[0][0:pc][:]
        scoreaverage=comp[1]
        image_shape=comp[2]
        mean=comp[4]
        X_people=comp[5]
        scorevariance=comp[6]
        combicomp=[]
        pcombicomp=[]
        ppcombicomp=[]
        fig=plt
        fig2=cv2
        fig3=plt
        plt.rcParams["figure.figsize"] = (20, 20)
        plt.rcParams["figure.dpi"] = 300
        scoreaverage_array = np.array(scoreaverage)
        scorevariance_array = np.array(scorevariance)
        randomsize=10
        
        for j in range(len(scoreaverage[0])):
            for k in range(randomsize):
                for i in range(len(components)):           
                    #ravdist=np.random.randint(scoreaverage_array[i,j]-3*(scorevariance_array[i,j]**0.5),scoreaverage_array[i,j]+3*(scorevariance_array[i,j]**0.5))
                    ravdist=np.random.normal(scoreaverage_array[i,j],scorevariance_array[i,j]**0.5,1)
                    pcombicomp.append(np.dot(np.expand_dims(ravdist,axis=0),np.expand_dims(components[i],axis=0)))
                spcombicomp=np.sum(pcombicomp,axis=0)
                ppcombicomp.append(spcombicomp)
                pcombicomp=[]

        pppcombicomp=ppcombicomp+mean
        ppppcombicomp=pppcombicomp.reshape(len(scoreaverage[0])*randomsize,144000)
        for i,j in enumerate(ppppcombicomp):
            reconst=j.reshape(image_shape)
            cv2.imwrite('{}/{}_test_{}_{}.jpg'.format(save_dir,"Data_augmentaion",i//randomsize+1,i%randomsize+1),reconst)
            plt.subplot(len(scoreaverage[0]),randomsize,i+1)            
            plt.imshow(reconst,cmap="gray")
            plt.title('CLASS {}'.format(i//randomsize+1),pad=2)
            plt.axis("off")   
        fig.savefig('{}/{}'.format(save_dir,"Data_Augmentaion"))
        plt.subplot(1,1,1)
        plt.imshow(pppcombicomp[0].reshape(image_shape),cmap="gray",vmax=150,vmin=50)
        plt.axis("off")   
        fig3.savefig('{}/{}'.format(save_dir,"sample_1"))

        
        '''for i in range(randomsize):  
            for j in range(len(scoreaverage[0])):
                ravdist=np.random.randint(scoreaverage_array[:,j]-3*(scorevariance_array[:,j]**0.5),scoreaverage_array[:,j]+3*(scorevariance_array[:,j]**0.5))          
                ravdist=np.expand_dims(ravdist,axis=0)
                combicomp1=np.matmul(ravdist,components)+mean
                combicomp.append(combicomp1)

        for i,j in enumerate(combicomp):            
            reconst=j.reshape(image_shape)
            cv2.imwrite('{}/{}_test_{}.jpg'.format(save_dir,"Data_augmentaion",i%7+1),reconst)
            plt.subplot(7,10,i+1)
            plt.title('class_{}'.format(i%7+1),fontsize=10)
            plt.imshow(reconst,cmap="gray",vmax=255,vmin=0)
            plt.axis("off")
           #np.savetxt('{}/{}'.format(save_dir,"array{}.csv".format(i)),reconst, fmt='%1.3f',delimiter=',')
        fig.savefig('{}/{}'.format(save_dir,"Data_Augmentaion"))'''
        
                        







          



        


           





 






    
    def eigen4test(a,b,c,d,test_image_dir):
        '''
        
        Parameters
        ----------
        a : str
            fluid type : ex)'CMC_crop2'
        b : str
            fluid type : ex)'CMC_crop2'
        c : str
            fluid type : ex)'CMC_crop2'
        d : str
            fluid type : ex)'CMC_crop2'
        e : str
            fluid type : ex)'CMC_crop2'
        f : str
            fluid type : ex)'CMC_crop2'
        g : str
            fluid type : ex)'CMC_crop2'
        test_image_dir : TYPE
            test_image for probability test dir. : ex) 'C:/Users/MCPL-JJ/eigen/new_test_image/220525_Carbo5PEO5_2skip_13multi_image.png'
    
        Returns
        -------
        None.
    
        '''
    
       
    
        with open('C:/Users/MCPL-JJ/eigen/micro_fluid.json', 'r') as j:
            json_data = json.load(j)
        base_dir = 'C:/Users/MCPL-JJ/eigen'
        CMC_dir = os.path.join(base_dir,'{}'.format(a))
        PEO_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        d_dir = os.path.join(base_dir,'{}'.format(d))
    
        
        CMC_list = os.listdir(CMC_dir)
        PEO_list = os.listdir(PEO_dir)
        c_list = os.listdir(c_dir)
        d_list = os.listdir(d_dir)
    
        
        CMC_images = np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0),axis=0)
        PEO_images = np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        d_images = np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0),axis=0)
    
        for i in range(1,len(CMC_list)):
            CMC_images = np.concatenate((CMC_images,np.expand_dims(cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_images = np.concatenate((PEO_images,np.expand_dims(cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        for i in range(1,len(d_list)):
            d_images = np.concatenate((d_images,np.expand_dims(cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0),axis=0)))
        images = np.concatenate((CMC_images,PEO_images,c_images,d_images),axis=0)
        
        CMC_data = np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[0])),0).flatten()),axis=0)
        PEO_data = np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        d_data = np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[0])),0).flatten()),axis=0)
    
        for i in range(1,len(CMC_list)):
            CMC_data = np.concatenate((CMC_data,np.expand_dims((cv2.imread(os.path.join(CMC_dir,'{}'.format(CMC_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(PEO_list)):
            PEO_data = np.concatenate((PEO_data,np.expand_dims((cv2.imread(os.path.join(PEO_dir,'{}'.format(PEO_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        for i in range(1,len(d_list)):
            d_data = np.concatenate((d_data,np.expand_dims((cv2.imread(os.path.join(d_dir,'{}'.format(d_list[i])),0).flatten()),axis=0)))
        data = np.concatenate((CMC_data,PEO_data,c_data,d_data),axis=0)    
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(CMC_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(PEO_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)
        target_3 = []
        for i in range(0,len(d_list)):
            target_3.append(3)
    
        target = target_0 + target_1 +target_2 + target_3 ##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c), '{}'.format(d)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            #ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 + target_3 ##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        X_people = data
        y_people = target
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
        
        test_image = cv2.imread(test_image_dir,0)
        test_image = test_image.reshape(1,-1)
        
        
        k_list = range(1,20)
        pca_c_list = range(1,100,10)
        
        
        fig1,axes1= plt.subplots(nrows=2,ncols=5,figsize=(35,14),dpi=100)
        # ax1.set_title("Accuracy(k,PCA_c)",fontsize=15)
        # ax1.set_ylabel("accuracy",fontsize=13)
        # ax1.set_xlabel("k",fontsize=13)
        # ax1.set_ylim([0,1.1])
        fig2,axes2= plt.subplots(nrows=2,ncols=5,figsize=(35,14),dpi=100)
        # ax2.set_title("probability",fontsize=15)
        # ax2.set_ylabel("probability",fontsize=13)
        # ax2.set_xlabel("k",fontsize=13)
        # ax2.set_ylim([0,1.1])
        
        ##data 누적
        pca_total = []
        knn_total = []
        a_proba_total = []
        b_proba_total = []
        c_proba_total = []
        d_proba_total = []
    
        
        for i in range(0,len(pca_c_list)): 
            proba_list = []
            proba_pca_list = []
            accuracies = []
            accuracies_pca = []
            pca = PCA(n_components=pca_c_list[i], whiten=True, random_state=0).fit(X_train)
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            test_image_pca = pca.transform(test_image)
            a_proba = []
            b_proba = []
            c_proba = []  
            d_proba = []
    
            for k in k_list:
                classifier = KNeighborsClassifier(n_neighbors = k)
                classifier.fit( X_train, y_train)    
                accuracies.append(classifier.score(X_test, y_test))
                proba = classifier.predict_proba(test_image)
                proba_list.append(proba)
                
                classifier_pca = KNeighborsClassifier(n_neighbors = k)
                classifier_pca.fit(X_train_pca, y_train)    
                accuracies_pca.append(classifier_pca.score(X_test_pca, y_test))
                proba_pca = classifier_pca.predict_proba(test_image_pca)
                proba_pca_list.append(proba_pca)
                
                pca_total.append(pca_c_list[i])
                knn_total.append(k)
                
                for j in range(0,7):
                    if j == 0 : 
                        result = float(proba_pca_list[k-1][0][j])
                        a_proba.append(result)
                    elif j == 1 :
                        result = float(proba_pca_list[k-1][0][j])
                        b_proba.append(result)
                    elif j == 2 :
                        result = float(proba_pca_list[k-1][0][j])
                        c_proba.append(result)
                    elif j == 3 :
                        result = float(proba_pca_list[k-1][0][j])
                        d_proba.append(result)
            a_proba_total = a_proba_total+a_proba
            b_proba_total = b_proba_total+b_proba
            c_proba_total = c_proba_total+c_proba
            d_proba_total = d_proba_total+d_proba
    
            
            
            a_predict_score = []
            b_predict_score = []
            c_predict_score = []
            d_predict_score = []
    
            for k in range(0,len(a_proba_total)):
                result = a_proba_total[k]*json_data["2ml/min"]["PEO5,Carbopol5"]
                a_predict_score.append(result)
            for k in range(0,len(a_proba_total)):
                result = b_proba_total[k]*json_data["2ml/min"]["PEO"]
                b_predict_score.append(result)       
            for k in range(0,len(a_proba_total)):
                result = c_proba_total[k]*json_data["2ml/min"]["PEO5,CMC5"]
                c_predict_score.append(result)
            for k in range(0,len(a_proba_total)):
                result = d_proba_total[k]*json_data["2ml/min"]["CMC5,Carbopol5"]
                d_predict_score.append(result)       
    
            score_sum = []
            for k in range(0,len(a_predict_score)):
                result = a_predict_score[k]+b_predict_score[k]+c_predict_score[k]+d_predict_score[k]
                score_sum.append(result)
                          
            
            if i < 5:
                ax1 = axes1[0,i]
                ax1.set_title("Accuracy(k,PCA_c)_{}".format(pca_c_list[i]),fontsize=15)
                ax1.set_ylabel("accuracy",fontsize=13)
                ax1.set_xlabel("k",fontsize=13)
                ax1.set_ylim([0,1.1])
                ax1.plot(k_list,accuracies_pca,"r-",label='{}'.format(i))
                ax1.legend(fontsize=12)
                
                ax2 = axes2[0,i]
                ax2.set_title("probability_{}".format(pca_c_list[i]),fontsize=15)
                ax2.set_ylabel("probability",fontsize=13)
                ax2.set_xlabel("k",fontsize=13)
                ax2.set_ylim([0,1.1])
                ax2.plot(k_list,a_proba,"r-",label='{}'.format(a))
                ax2.legend(fontsize=12)   
                ax2.plot(k_list,b_proba,"b-",label='{}'.format(b))
                ax2.legend(fontsize=12)   
                ax2.plot(k_list,c_proba,"g-",label='{}'.format(c))
                ax2.legend(fontsize=12)   
                ax2.plot(k_list,d_proba,"c-",label='{}'.format(d))
                ax2.legend(fontsize=12)   
      
            else:
                ax1 = axes1[1,i-5]
                ax1.set_title("Accuracy(k,PCA_c)_{}".format(pca_c_list[i]),fontsize=15)
                ax1.set_ylabel("accuracy",fontsize=13)
                ax1.set_xlabel("k",fontsize=13)
                ax1.set_ylim([0,1.1])    
                ax1.plot(k_list,accuracies_pca,"r-",label='{}'.format(i))
                ax1.legend(fontsize=12)    
        
                ax2 = axes2[1,i-5]
                ax2.set_title("probability_{}".format(pca_c_list[i]),fontsize=15)
                ax2.set_ylabel("probability",fontsize=13)
                ax2.set_xlabel("k",fontsize=13)
                ax2.set_ylim([0,1.1])        
                ax2.plot(k_list,a_proba,"r-",label='{}'.format(a))
                ax2.legend(fontsize=12)   
                ax2.plot(k_list,b_proba,"b-",label='{}'.format(b))
                ax2.legend(fontsize=12)   
                ax2.plot(k_list,c_proba,"g-",label='{}'.format(c))
                ax2.legend(fontsize=12)   
                ax2.plot(k_list,d_proba,"c-",label='{}'.format(d))
                ax2.legend(fontsize=12)   
            
        ##엑셀로 probability 및 accuracy 취합 (knn 및 proba에 따라)
        data = {'pca_c':pca_total, 'knn_k':knn_total, 'a_proba':a_proba_total, 'b_proba':b_proba_total, 'c_proba':c_proba_total, 'd_proba':d_proba_total}
        data = pd.DataFrame(data)
        savenumber = test_image_dir.replace(base_dir,"")
        savenumber = savenumber.replace("/","_")
        savenumber = savenumber.replace(".png","")
        savenumber = savenumber.replace("multi_image","")
        data.to_excel(excel_writer= base_dir+'/'+'{}.xlsx'.format(savenumber),sheet_name='pca,knn')            
        data2 = {'pca_c':pca_total, 'knn_k':knn_total, 'a_proba':a_predict_score, 'b_proba':b_predict_score, 'c_proba':c_predict_score, 'd_proba':d_predict_score,'sum':score_sum}
        data2 = pd.DataFrame(data2)
        with pd.ExcelWriter(base_dir+'/'+'{}.xlsx'.format(savenumber),engine='openpyxl',mode='a') as writer:
            data2.to_excel(writer,sheet_name='score')
        # data2.to_excel(excel_writer= base_dir+'/'+'{}.xlsx'.format(savenumber),sheet_name='score')      
        # ax1.plot(k_list,accuracies_pca,"r-",label='{}'.format(pca_c))
        # ax1.legend(fontsize=12)
        # ax2.plot(k_list,a_proba,"r-",label='{}'.format(a))
        # ax2.legend(fontsize=12)   
        # ax2.plot(k_list,b_proba,"b-",label='{}'.format(b))
        # ax2.legend(fontsize=12)   
        # ax2.plot(k_list,c_proba,"g-",label='{}'.format(c))
        # ax2.legend(fontsize=12)   
        

    
    def eigen3test(self,a,b,c,test,pc1,pc2,pc3,pca_c=2):

        import cv2
        import numpy as np
        import os
        import scipy.optimize as optimize
        import matplotlib.pyplot as plt
        from mcplexpt.caber.dos import DoSCaBERExperiment
        from mcplexpt.testing import get_samples_path
        from sklearn.preprocessing import StandardScaler  ##StandardScaler :  평균0,분산1로 표준화 해준다.
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        from mpl_toolkits.mplot3d import Axes3D
        import json
        import pandas as pd
        import seaborn as sns


        '''
        
    
        Parameters
        ----------
        a : str
            분류유체1의 이름 (사진이 저장된 파일명과 같아야함)
        b : str
            분류유체2의 이름 (사진이 저장된 파일명과 같아야함)
        c : str
            분류유체3의 이름 (사진이 저장된 파일명과 같아야함)
        pca_c : int, optional
            분류시 사용 할 주성분의 갯수. The default is 10.
    
        Returns
        -------
        None.
    
        '''
        filename=a[7:]+b[7:]+c[7:]+test[7:]
        base_dir = '/home/minhyukim/PCA/'
        save_dir = base_dir+'PCA_result/'+filename


        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        except OSError:
            print ('Error: Creating directory. ' +  save_dir)


        a_dir = os.path.join(base_dir,'{}'.format(a))
        b_dir = os.path.join(base_dir,'{}'.format(b))
        c_dir = os.path.join(base_dir,'{}'.format(c))
        test_dir = os.path.join(base_dir,'{}'.format(test))


        
        a_list = os.listdir(a_dir)
        b_list = os.listdir(b_dir)
        c_list = os.listdir(c_dir)
        test_list=os.listdir(test_dir)
        
        a_images = np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0),axis=0)
        b_images = np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0),axis=0)
        c_images = np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0),axis=0)
        test_images= np.expand_dims(cv2.imread(os.path.join(test_dir,'{}'.format(test_list[0])),0),axis=0)

        for i in range(1,len(a_list)):
            a_images = np.concatenate((a_images,np.expand_dims(cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0),axis=0)))
        for i in range(1,len(b_list)):
            b_images = np.concatenate((b_images,np.expand_dims(cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0),axis=0)))
        for i in range(1,len(c_list)):
            c_images = np.concatenate((c_images,np.expand_dims(cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0),axis=0)))
        
        for i in range(1,len(test_list)):
            test_images = np.concatenate((test_images,np.expand_dims(cv2.imread(os.path.join(test_dir,'{}'.format(test_list[i])),0),axis=0)))


        images = np.concatenate((a_images,b_images,c_images),axis=0)
        
        a_data = np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[0])),0).flatten()),axis=0)
        b_data = np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[0])),0).flatten()),axis=0)
        c_data = np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[0])),0).flatten()),axis=0)
        test_data = np.expand_dims((cv2.imread(os.path.join(test_dir,'{}'.format(test_list[0])),0).flatten()),axis=0)

        for i in range(1,len(a_list)):
            a_data = np.concatenate((a_data,np.expand_dims((cv2.imread(os.path.join(a_dir,'{}'.format(a_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(b_list)):
            b_data = np.concatenate((b_data,np.expand_dims((cv2.imread(os.path.join(b_dir,'{}'.format(b_list[i])),0).flatten()),axis=0)))
        for i in range(1,len(c_list)):
            c_data = np.concatenate((c_data,np.expand_dims((cv2.imread(os.path.join(c_dir,'{}'.format(c_list[i])),0).flatten()),axis=0))) 
        data = np.concatenate((a_data,b_data,c_data),axis=0)   

        for i in range(1,len(test_list)):
            test_data = np.concatenate((test_data,np.expand_dims((cv2.imread(os.path.join(test_dir,'{}'.format(test_list[i])),0).flatten()),axis=0))) 
 
    
        ## CMC = 0, PEO = 1 로 넘버링
        target_0 = []
        for i in range(0,len(a_list)):
            target_0.append(0)
        target_1 = []
        for i in range(0,len(b_list)):
            target_1.append(1)
        target_2 = []
        for i in range(0,len(c_list)):
            target_2.append(2)

        target = target_0 + target_1 +target_2 
        target = np.array(target)
    
        target_names = ['{}'.format(a),'{}'.format(b),'{}'.format(c)]
    
        image_shape = images[0].shape ## image는 실제 이미지샘플 1개 의미함 (여기서 people.은 이미지 보관 장소?)
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()}) ##이미지 확인
        for target, image, ax in zip(target,images, axes.ravel()):
            #ax.imshow(image)
            ax.set_title(target_names[target])
        plt.show()     
    
        ##
        print("images.shape: {}".format(images.shape))
        print("클래스의 개수: {}".format(len(target_names)))
        
        ##target 이 자꾸1로 변경되서 다시 추가함
        target = target_0 + target_1 +target_2 ##순서 바뀌면 안됨. 앞의 이미지와 매칭되는 넘버링 이기때문에
        target = np.array(target)
        
        
        # mask = np.zeros(target.shape, dtype=np.bool)
        # for i in np.unique(target):
        #     mask[np.where(target== i)[0][:]] = 1
        
        # X_people = data[mask]
        # y_people = target[mask]
        
        # X_people = X_people / 255.
        X_people = data
        y_people = target
        datavar=np.var(X_people,0)
        varmax=np.max(datavar) 
        
        ##
        X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
    
        '''
        ##향후에 k값 변경 필요시 사용
        k_list = range(1,101)
        accuracies = []
        for k in k_list:
          classifier = KNeighborsClassifier(n_neighbors = k)
          classifier.fit(training_data, training_labels)
          accuracies.append(classifier.score(validation_data, validation_labels))
        plt.plot(k_list, accuracies)
        plt.xlabel("k")
        plt.ylabel("Validation Accuracy")
        plt.title("Breast Cancer Classifier Accuracy")
        plt.show()
        ''' 
        knn = KNeighborsClassifier(n_neighbors=60)
        knn.fit(X_train, y_train)
        
        ##
        pca = PCA(n_components=pca_c, whiten=False, random_state=0).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        #X_people_pca = pca.transform(X_people)
        #Test_pca=pca.transform(test_data)

        
        
        ##
        fig, ax = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': (0,200,400), 'yticks': (360,180,0)})
        for i, (component, ax) in enumerate(zip(pca.components_, ax.ravel())):
            #ax.imshow(component.reshape(image_shape), cmap='viridis')
            ax.set_title("PC {}".format(i+1))

        fig.savefig('{}/{}'.format(save_dir,"Principal component"))

        Fluid=['F(1)','F(2)','F(3)','F(4)','F(5)','F(6)','F(7)']
        Carboref=[1,0,0,0.5,0  ,0.5,0.33]
        CMCref=  [0,1,0,0.5,0.5,0,  0.33]
        PEOref=  [0,0,1,0  ,0.5 ,0.5,0.33]

        proba_list=[]
      
        k_list = range(1,10)

        classifier = KNeighborsClassifier(n_neighbors = 60)
        classifier.fit(X_train_pca, y_train)
        proba = classifier.predict_proba(X_test_pca)
        print(y_test)
        proba_list.append(proba)
        fig10,ax10 = plt.subplots()

        ax10.bar(Fluid,Carboref,color='r',width=0.2)
        ax10.bar(Fluid,CMCref,color='g', width=0.2,bottom=Carboref)
        ax10.bar(Fluid,PEOref,color='b', width=0.2,bottom=[Carboref[i]+CMCref[i] for i in range(len(Carboref))])

        proba_list=np.array(proba_list)
        print(proba_list[0])
        ax10.bar([ i + 0.4 for i in range(len(Fluid))],np.average(proba_list[0][:,0]),color='r',width=0.2)
        ax10.bar([ i + 0.4 for i in range(len(Fluid))],np.average(proba_list[0][:,1]),bottom=np.average(proba_list[0][:,0]),color='g',width=0.2)
        ax10.bar([ i + 0.4 for i in range(len(Fluid))],np.average(proba_list[0][:,2]),bottom=np.average(proba_list[0][:,0])+np.average(proba_list[0][:,1]),color='b',width=0.2)
        
        fig10.savefig('{}/{}'.format(save_dir,"Probability_weight"))
        

        
            

      
    
    
#3D SCATTER            

        
                        
        '''fig2 = plt.figure(figsize=(15,15))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        ax2.set_xlabel('Principal Component'+' '+str(pc1+1), fontsize = 15)
        ax2.set_ylabel('Principal Component'+' '+str(pc2+1), fontsize = 15)
        ax2.set_zlabel('Principal Component'+' '+str(pc3+1), fontsize = 15)
        ax2.set_title('3 Component PCA', fontsize = 20)
        
        colors = ["b","g","r"]
        labels = [0,1,2]
        label2 = [a[7:],b[7:],c[7:]]
        
        
        for label,color,label2 in zip(labels,colors,label2):                                    
                            
            ax2.scatter(X_train_pca[np.where(label==y_train),pc1], X_train_pca[np.where(label==y_train),pc2], X_train_pca[np.where(label==y_train),pc3],label= label2, c= color, s = 30)              
            ax2.grid()
            ax2.legend()
        fig2.savefig('{}/{}'.format(save_dir,"3D_Scatter"+'_{}_{}_{}'.format(pc1,pc2,pc3)))
            
            
            
            
#3D SCATTER(Labeling one by one)     
        
        
        
        fig3 = plt.figure(figsize=(15,15))
        ax3 = fig3.add_subplot(111, projection='3d')
        
        ax3.set_xlabel('Principal Component'+' '+str(pc1+1), fontsize = 15)
        ax3.set_ylabel('Principal Component'+' '+str(pc2+1), fontsize = 15)
        ax3.set_zlabel('Principal Component'+' '+str(pc3+1), fontsize = 15)
        ax3.set_title('3 Component PCA', fontsize = 20)
        
        flag1=0
        flag2=0
        flag3=0

        
        for i in range(len(y_people)):       
        
                if y_people[i] == 0:
                    flag1 = flag1 + 1
                    label2='$'+a_list[flag1-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag1,'flag1')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='b',marker= label3,s = 6000)
                            
                elif y_people[i] == 1:
                    flag2 = flag2 + 1
                    label2='$'+b_list[flag2-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag2,'flag2')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='g',marker= label3,s = 6000)


                else:
                    flag3 = flag3 + 1
                    label2='$'+c_list[flag3-1][7:-15]+'$'
                    label3=label2.replace('_',' ')
                    print(flag3,'flag3')
                    ax3.scatter(X_people_pca[i,pc1], X_people_pca[i,pc2], X_people_pca[i,pc3],c='r',marker= label3,s = 6000)
                print(i,'i')

        fig3.savefig('{}/{}'.format(save_dir,"3D_Scatter_Labelling"))
        samplenumber={'a':len(a_list),'b':len(b_list),'c':len(c_list)}
        data = {'pc1':X_people_pca[:,0], 'pc2':X_people_pca[:,1], 'pc3':X_people_pca[:,2], 'pc4':X_people_pca[:,3], 'pc5':X_people_pca[:,4], 'pc6':X_people_pca[:,5],'pc7':X_people_pca[:,6],'pc8':X_people_pca[:,7],'pc9':X_people_pca[:,8],'pc10':X_people_pca[:,9]}
        data2= samplenumber

        list_list=[a_list,b_list,c_list]
        scoreaverage=[]
        
        print(len(a_list),len(b_list),len(c_list))
        
        for j in range(0,10):
            averagelist=[]
            flag=0
            for i in range(0,3):
                if i == 0:
                    average=sum(X_people_pca[0:len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    flag=len(list_list[i])                    
                else:                    
                    average=sum(X_people_pca[flag:flag+len(list_list[i]),j])/len(list_list[i])
                    averagelist.append(average)
                    flag=flag+len(list_list[i])
                    
            scoreaverage.append(averagelist)
        print(scoreaverage)
        data3=scoreaverage


        data = pd.DataFrame(data)
        data2= pd.DataFrame(data2,index=[0])
        data3= pd.DataFrame(data3)

        data.to_excel(excel_writer= save_dir+'/{}.xlsx'.format('PC',sheet_name='Principal component'))
        exfilename= save_dir+'/PC.xlsx'
        with pd.ExcelWriter(str(exfilename), mode='a', engine='openpyxl') as writer:
            data2.to_excel(writer,sheet_name='sample number')
            data3.to_excel(writer,sheet_name='score averge' )'''
        
#Stack-up plot
        components=pca.components_
        y_axis=np.cumsum(np.multiply(components[pc1,:],components[pc1,:]))
        x_axis=range(0,144000)
        fig5,ax5=plt.subplots(figsize=(12,12))
        ax5.plot(x_axis,y_axis)
        fig5.savefig('{}/{}'.format(save_dir,"Stackup".format(pc1)))



        fig6,ax6=plt.subplots(figsize=(12,12))
        gradyvalue=[]
        gradxvalue=range(0,143800,200)
        for i in gradxvalue:
            grad1=(y_axis[i+200]-y_axis[i])/(x_axis[i+200]-x_axis[i])
            gradyvalue.append(grad1)
            
            
        ax6.plot(gradxvalue,gradyvalue)
        fig6.savefig('{}/{}'.format(save_dir,"Stackup_grad".format(pc1)))

    #Circle of correlation
     
        components=pca.components_
        ev=pca.explained_variance_

        loading=components.T * np.sqrt(ev)
        fig7,ax7=plt.subplots(figsize=(12,12))
        an = np.linspace(0, 2 * np.pi, 100)
        ax7.plot(np.cos(an), np.sin(an),'r')
        
        ax7.scatter(loading[:,0]/np.sqrt(varmax),loading[:,1]/np.sqrt(varmax))
        ax7.set_title('Variable factor map')
        fig7.savefig('{}/{}'.format(save_dir,"CC_{}_{}".format(pc1,pc2)))

    #Important Pixels Marking(PCs)


        fig8, ax8 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        for i,(component,ax) in enumerate(zip(components,ax8.ravel())):
           sortpc=np.argsort(component)[::-1]
           percent=sortpc[:int(0.05*(len(sortpc)))]
           normspc=component*255/np.max(component)
           expandpc=np.expand_dims(normspc,axis=1)
           expandpc=np.expand_dims(expandpc,axis=2)
           stackpc=np.concatenate((expandpc,expandpc),axis=2)
           stackpc=np.concatenate((stackpc,expandpc),axis=2)
           stackpc[percent,:,0]=255
           stackpc[percent,:,1]=0
           stackpc[percent,:,2]=0
           rsp_stack=np.reshape(stackpc,(360,400,3))
          #ax.imshow(rsp_stack,cmap='viridis')
           ax.set_title("Important Pixels PC{}".format(i+1))
        fig8.savefig('{}/{}'.format(save_dir,"Important Pixels"))

    #Important Pixels Marking(Pixel Ranking)
        fig9, ax9 = plt.subplots(2,5, figsize=(15,8), subplot_kw={'xticks': [0,200,400], 'yticks': [360,180,0]})
        for i,(ax) in enumerate(ax9.ravel()):
           p1=components[0]
           sortpc=np.argsort(p1)[::-1]
           percent=sortpc[:int(0.05*(i+1)*(len(sortpc)))]
           normspc=p1*255/np.max(p1)
           expandpc=np.expand_dims(normspc,axis=1)
           expandpc=np.expand_dims(expandpc,axis=2)
           stackpc=np.concatenate((expandpc,expandpc),axis=2)
           stackpc=np.concatenate((stackpc,expandpc),axis=2)
           stackpc[percent,:,0]=255
           stackpc[percent,:,1]=0
           stackpc[percent,:,2]=0
           rsp_stack=np.reshape(stackpc,(360,400,3))
           #ax.imshow(rsp_stack,cmap='viridis')
           ax.set_title("Important Pixels Rank {}%".format((i+1)*5))
        fig9.savefig('{}/{}'.format(save_dir,"Important Pixels(Rank)")) 


                
        
#%%        
        

        '''
            
            Parameters
            ----------
            a : str
                fluid type : ex)'CMC_crop2'
            b : str
                fluid type : ex)'CMC_crop2'
            c : str
                fluid type : ex)'CMC_crop2'
            d : str
                fluid type : ex)'CMC_crop2'
            e : str
                fluid type : ex)'CMC_crop2'
            f : str
                fluid type : ex)'CMC_crop2'
            g : str
                fluid type : ex)'CMC_crop2'
            test_image_dir : TYPE
                test_image for probability test dir. : ex) 'C:/Users/MCPL-JJ/eigen/new_test_image/220525_Carbo5PEO5_2skip_13multi_image.png'
        
            Returns
            -------
            None.
        
        '''


            
            
            
            
            
            
            
