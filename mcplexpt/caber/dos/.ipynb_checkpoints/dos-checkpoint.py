"""
DoS CaBER 실험 이미지를 분석하기 위한 모듈입니다.

"""
import cv2, sys
import numpy as np
import cupy as cp
import scipy.optimize as optimize
from mcplexpt.core import ExperimentTIFF


class DoSCaBERExperiment(ExperimentTIFF):
    """
    DoS CaBER 실험 결과 TIFF 이미지 파일을 추상화하는 클래스입니다.

    Examples
    ========

    >>> from mcplexpt.caber.dos import DoSCaBERExperiment
    >>> from mcplexpt.testing import get_samples_path
    >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
    >>> expt = DoSCaBERExperiment(path)

    """
    

    def capbridge_start(self, image): ## 설명 Capillarty thinning 측정시작부를 판단하기 위한 코드

        image_check = self.get_nth_image(-1)
        image_check = image_check.download()
        image_c=cv2.Canny(image_check,50,220)    
        yss, xss = np.where(image_c == 255)
      
        thick_l =[]
        row_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 and thick < int(sum(thick_l)/len(thick_l)-10) : ## 노즐 두께에 따라 310이 아닐수 있기때문에 변경필요.
                row_l.append(row)
                break    
        nozzle_thick = int(sum(thick_l)/len(thick_l))
        
        row_roi = row_l[0]
        image=image.download()
        h,w = image.shape
        roi = image[int(row_roi):int(row_roi+0.2*nozzle_thick),0:w]    
        
        roi_c = cv2.Canny(roi,50,220)
        ys,xs = np.where(roi_c == 255)
        thick_l = []
        for y in set(ys):
            edge_xss = xs[np.where(ys == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            thick_l.append(thick)
        minus_slope = []
        plus_slope = []
        for i in range(len(thick_l)):
            if i>=1:
                if thick_l[i]-thick_l[i-1] == 0:
                    pass
                elif thick_l[i]-thick_l[i-1] < 0 :
                    minus_slope.append(i)
                else:
                    plus_slope.append(i)
            elif i == 0:
                if thick_l[i+1]-thick_l[i] == 0:
                    pass
                elif thick_l[i+1]-thick_l[i] < 0 :
                    minus_slope.append(i)
                else:
                    plus_slope.append(i)
                            
        if len(plus_slope) >= len(minus_slope) :
            return False
        elif len(plus_slope) < len(minus_slope):
            return True

    def capbridge_broken(self, image):
        """
        이미지에서 capillary bridge가 끊어져 있는지 여부를 판단합니다.

        Parameters
        ==========

        image : np.ndarray
            실험에서의 이미지입니다.

        Returns
        =======

        bool

        Examples
        ========

        Capillary bridge가 존재할 경우 False를 반환합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotfluilib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path
            >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
            >>> expt = DoSCaBERExperiment(path)
            >>> img1 = expt.get_nth_image(0)
            >>> plt.imshow(img1, cmap='gray') # doctest: +SKIP
            >>> expt.capbridge_broken(img1) # doctest: +SKIP
            False

        .. plot::
            :include-source:
            :context: close-figs

            >>> img2 = expt.get_nth_image(-1)
            >>> plt.imshow(img2, cmap='gray') # doctest: +SKIP
            >>> expt.capbridge_broken(img2) # doctest: +SKIP
            True

        """
        image_check = self.get_nth_image(-1)
        image_check = image_check.download()
        image_c=cv2.Canny(image_check,50,220)     
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        row_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 and thick < int(sum(thick_l)/len(thick_l)-10) : ## 노즐 두께에 따라 310이 아닐수 있기때문에 변경필요.
                row_l.append(row)
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))
        row_roi = row_l[0]
        image=image.download()
        h,w = image.shape
        roi_image = image[0:int(row_roi+1.5*nozzle_thick+10),0:w] ## Neck의 하단부는 roi에서 잘려나감. 조금 더 넓은범위 확인할 수 있으면서 노이즈끼지 않도록 수정필요.

        ret,thresh_image = cv2.threshold(roi_image,130,255,cv2.THRESH_BINARY)
        
        y_w, x_w = np.where(thresh_image==255)
        y_b, x_b = np.where(thresh_image==0)
        
        y_w_row = set(y_w)
        y_b_row = set(y_b)
        only_w_y_row = set.difference(y_w_row,y_b_row)
        h,w = thresh_image.shape
        
        if len(only_w_y_row) == 0:
            return False
        elif len(only_w_y_row) != 0 and max(only_w_y_row) != h and min(only_w_y_row) != 0:            
            return True

    def width_average_fixedroi_modified(self,image):
        """
        Frame(time)에 관계없이 고정된 ROI 내부에서 Neck의 Average Width를 구합니다.

        Parameters
        ==========

        image : np.ndarray
            실험에서의 이미지입니다.

        Returns
        =======

        avg_width : int
            고정된 ROI 영역내의 10개 Row에 대한 Neck의 Wdith 평균값 입니다.

        Examples
        ========

        아래 그림과 같이 Neck이 형성된 경우 Average Width를 반환합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path
            >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
            >>> expt = DoSCaBERExperiment(path)
            >>> img1 = expt.get_nth_image(-1)
            >>> plt.imshow(img1, cmap='gray') # doctest: +SKIP

        >>> expt.width_average_fixedroi(img1)
        3

        아래 그림과 같이 Neck이 형성되지 않은 경우 DoSCaBERError 를 반환합니다.

        .. plot::
            :include-source:
            :context: close-figs

            >>> img2 = expt.get_nth_image(0)
            >>> plt.imshow(img2, cmap='gray') # doctest: +SKIP         

        >>> expt.width_average_fixedroi(img2)
        Traceback (most recent call last):
        ...
        DoSCaBERError: Neck is not formed yet  

        Raises
        ======

        ValueError
            Neck이 형성되지 않았습니다

        """  
        image_check = self.get_nth_image(-1)
        image_c = cv2.cuda.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        row_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 and thick < int(sum(thick_l)/len(thick_l)-10) : ## 노즐 두께에 따라 310이 아닐수 있기때문에 변경필요.
                row_l.append(row)
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))
        row_roi = row_l[0]
        
        h,w = image.shape
        roi = image[int(row_roi+0.65*nozzle_thick):int(row_roi+0.65*nozzle_thick+10),0:w] # roi 설정 기준은 1Di 위치에서 측정하는것을 기준으로 함.(명확한 근거는 없고, 영상 이미지 참고 및 H=3Di중 Substrate상에 형성되는 Drop의 높이를 고려하여 설정)
        roi_img = roi.copy()
        roi_img_canny = cv2.Canny(roi_img,50,220)
        ys, xs = np.where(roi_img_canny==255)
        row_row = []
        row_thick = [] # item=(row, thickness)
        for y in set(ys):
            edge_xs = xs[np.where(ys==y)]
            row, width = (y, edge_xs[-1]-edge_xs[0])
            row_thick.append(width)
            row_row.append(row)
        if len(row_row) ==0:
            raise DoSCaBERError('Neck is not formed yet')
        else:
            avg_width = (sum(row_thick) / len(row_row))
            return avg_width
    
    def width_average_minimumroi_moving(self, image):
        """
        이미지에서 Neck 영역의 minimum wdith가 생기는 row를 찾고, 
        측정한 width 중 작은값순으로 총 10개 값에 대한 Average width를 구합니다. 
        다만, 유체에 따라 (특히 viscoelstic한 거동을 보여서 neck이 1자형으로 형성 시) 그 값이 흔들릴 가능성이 있고,
        이를 개선하기위해, moving average 개념을 도입하여 raw data의 frame 3개의 width를 평균으로하여 data를 정리하였음.

        Parameters
        ==========

        image : np.ndarray
            실험에서의 이미지입니다.

        Returns
        =======

        avg_width : int
            고정된 ROI 영역내의 10개 Row에 대한 Neck의 Wdith 평균값 입니다.

        Examples
        ========

        아래 그림과 같이 Neck 이 형성된 경우 Average Width를 반환합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path
            >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
            >>> expt = DoSCaBERExperiment(path)
            >>> img1 = expt.get_nth_image(-1)
            >>> plt.imshow(img1, cmap='gray') # doctest: +SKIP

        >>> expt.width_average_minimumroi_moving(img1)
        1

        아래 그림과 같이 Neck이 형성되지 않은 경우 DoSCaBERError 를 반환합니다.

        .. plot::
            :include-source:
            :context: close-figs

            >>> img2 = expt.get_nth_image(0)
            >>> plt.imshow(img2, cmap='gray') # doctest: +SKIP         

        >>> expt.width_average_minimumroi(img2)
        Traceback (most recent call last):
        ...
        DoSCaBERError: Neck is not formed yet  

        Raises
        ======

        ValueError
            Neck이 형성되지 않았습니다

        """
        image_check = self.get_nth_image(-1)
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 :
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))

        image_canny = cv2.Canny(image,50,220)
        ys,xs = np.where(image_canny==255)
        h, w = image_canny.shape
        roi = image_canny[0:h,xs[0]:xs[1]+1]

        
        row_row = []
        row_thick = [] # item=(row, thickness)

        for y in set(ys):
            edge_xs = xs[np.where(ys==y)]
            row, width = (y, edge_xs[-1]-edge_xs[0])
            if width < nozzle_thick:
                row_thick.append(width)
                row_row.append(row)           


        t_roi = np.transpose(roi)
        
        ybt = np.where(t_roi[0] == 255)
        ybt1 = list(ybt)
        ybt_list = ybt1[0]
        break_y = max(ybt_list)
        
        minus = []
        try:
            for i in row_row:
                if i > break_y:
                    minus.append(i)
            
            row_thick_real =row_thick[0:len(row_row)-len(minus)]
            row_thick_real_moving_average = []
            for i in range(len(row_thick_real)):
                if i < 2 : 
                   row_thick_real_moving_average.append(row_thick_real[i]) 
                else:
                    moving_average = (row_thick_real[i]+row_thick_real[i-1]+row_thick_real[i-2])/3
                    row_thick_real_moving_average.append(moving_average)
            
            row_thick_real_moving_average.sort()
            avg_width_find_list = row_thick_real_moving_average[0:9]
            avg_width = (sum(avg_width_find_list)/len(avg_width_find_list))
            
            if len(row_thick_real) < 50 : ## 실제로는 충분히 neck이 형성되지 않았거나 화면 에러의에 한 측정결과는 필터링하기위해 입력. (필터기준링 은 추후 재고려 필요.)
                raise DoSCaBERError('Neck is not formed yet')
            else:    
                return avg_width
        except ZeroDivisionError: #일부 이미지 초기구간에서 ZeroError 발생해서 추가함.. 상세원인은 추가적인 고민 필요.
            pass

    def width_average_minimumroi_modified(self,image): ## 후반부 전반적인 수정필요.

        """
        이미지에서 Neck 영역의 minimum wdith가 생기는 row를 찾고, 
        해당 row를 기준으로 전,후 4개 row를 포함한 총 9개 row에 대한 Average width를 구합니다. 
        다만, 유체에 따라 (특히 viscoelstic한 거동을 보여서 neck이 1자형으로 형성 시) 그 값이 흔들릴 가능성이 있고,
        이럴 경우에는 Moving average와 같은 기법을 적용하여, 값을 Smoothing 시키는 방법으로 업그레이드 할 필요성 있어보입니다.

        Parameters
        ==========

        image : np.ndarray
            실험에서의 이미지입니다.

        Returns
        =======

        avg_width : int
            고정된 ROI 영역내의 10개 Row에 대한 Neck의 Wdith 평균값 입니다.

        Examples
        ========

        아래 그림과 같이 Neck 이 형성된 경우 Average Width를 반환합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path
            >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
            >>> expt = DoSCaBERExperiment(path)
            >>> img1 = expt.get_nth_image(-1)
            >>> plt.imshow(img1, cmap='gray') # doctest: +SKIP

        >>> expt.width_average_minimumroi(img1)
        2

        아래 그림과 같이 Neck이 형성되지 않은 경우 DoSCaBERError 를 반환합니다.

        .. plot::
            :include-source:
            :context: close-figs

            >>> img2 = expt.get_nth_image(0)
            >>> plt.imshow(img2, cmap='gray') # doctest: +SKIP         

        >>> expt.width_average_minimumroi(img2)
        Traceback (most recent call last):
        ...
        DoSCaBERError: Neck is not formed yet  

        Raises
        ======

        ValueError
            Neck이 형성되지 않았습니다

        """
        image_check = self.get_nth_image(-1)
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 :
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))

        image_canny = cv2.Canny(image,50,220)
        ys,xs = np.where(image_canny==255)
        h, w = image_canny.shape
        roi = image_canny[0:h,xs[0]:xs[1]+1]

        
        row_row = []
        row_thick = [] # item=(row, thickness)

        for y in set(ys):
            edge_xs = xs[np.where(ys==y)]
            row, width = (y, edge_xs[-1]-edge_xs[0])
            if width < nozzle_thick:
                row_thick.append(width)
                row_row.append(row)           


        t_roi = np.transpose(roi)
        
        ybt = np.where(t_roi[0] == 255)
        ybt1 = list(ybt)
        ybt_list = ybt1[0]
        break_y = max(ybt_list)
        
        minus = []
        try:
            for i in row_row:
                if i > break_y:
                    minus.append(i)
            
            row_thick_real =row_thick[0:len(row_row)-len(minus)]     
            row_thick_real_moving_average = []
            for i in range(len(row_thick_real)):
                if i >= 9:
                    moving_average = (row_thick_real[i]+row_thick_real[i-1]+row_thick_real[i-2]+row_thick_real[i-3]+row_thick_real[i-4]+row_thick_real[i-5]+row_thick_real[i-6]+row_thick_real[i-7]+row_thick_real[i-8]+row_thick_real[i-9])/10
                    row_thick_real_moving_average.append(moving_average)
                else:
                    moving_average = row_thick_real[i]
                    row_thick_real_moving_average.append(moving_average)
                    
            if len(row_thick_real) < 50 : ## 실제로는 충분히 neck이 형성되지 않았거나 화면 에러의에 한 측정결과는 필터링하기위해 입력. (필터기준링 은 추후 재고려 필요.)
                raise DoSCaBERError('Neck is not formed yet')
            else:    
                return min(row_thick_real_moving_average)
        except ZeroDivisionError: #일부 이미지 초기구간에서 ZeroError 발생해서 추가함.. 상세원인은 추가적인 고민 필요.
            pass

    def Image_Radius_Measure_temp(self,tifname,savenumber,fps):
        """
        측정영상(tif file)을 Frame 별로 분석하여 측정시간/ Break시간/ Wdith 값을 엑셀로 추출합니다.
        fps에 따라 code 정보 변경해야 합니다. (기본 10000 fps로 설정)

        Parameters
        ==========

        tifname : str
            측정이미지의 경로로, "filename.tif (or .tiff)" 형태로 입력합니다.

        savenumber : int
            data가 저장될 엑셀의 파일명으로, 20210723 과 같이 숫자를 입력합니다. (추후 수정 계획)

        Returns
        =======
        없음
        현재 경로에 data_savenumber.xlsx 파일을 형성합니다.

        Examples
        ========

        DoS CaBER 측정이미지를 통해, 시간(frame)에 따른 Neck의 Width 및 Break여부를 엑셀data로 추출합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path

        >>> expt.Dos_CaBER_fixed_min("sample_250fps.tiff",20210723) # doctest: +SKIP
        data_20210723.xlsx 파일 형성 # doctest: +SKIP

        Raises
        ======

        """
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from mcplexpt.caber.dos import DoSCaBERExperiment
        from mcplexpt.testing import get_samples_path
        path = get_samples_path("caber", "dos", tifname)
        expt = DoSCaBERExperiment(path)

        directory = os.getcwd()+'/relaxationtime/'
        try:
            if not os.path.exists(directory+savenumber):
                os.makedirs(directory+savenumber)
        except OSError:
            print ('Error: Creating directory. ' +  directory+savenumber)

        ret0=[]
        for i in range(0,10000000):
            try:
                image = expt.get_nth_image(i)
                ret0.append(i)
            except IndexError:
                break
        frame_number = len(ret0)

        ret_s = []
        for i in range(0,frame_number):
            image = expt.get_nth_image(i)
            result = expt.capbridge_start(image)
            ret_s.append(result)
        start_f = 0
        ret = []
        for i in range(0,frame_number):
            image = expt.get_nth_image(i)
            result = expt.capbridge_broken(image)
            ret.append(result)
        break_f = frame_number

        ##이미지 저장
        '''if break_f>20 and frame_number-break_f> 10:
            for i in range(break_f-20,break_f+10):
                image = expt.get_nth_image(i)
                cv2.imwrite(directory+savenumber+'/'+savenumber+'_image{}.png'.format(i),image)
        else: 
            for i in range(break_f-10,break_f+5):
                image = expt.get_nth_image(i)
                cv2.imwrite(directory+savenumber+'/'+savenumber+'_image{}.png'.format(i),image)'''
        
        ret1 = []
        for i in range(0,frame_number):
            try:
                if i>=start_f and i<break_f:
                    image = expt.get_nth_image(i)
                    width = expt.width_average_fixedroi_modified(image) 
                    ret1.append(width)
                elif i<start_f:
                    start_index = 'None'
                    ret1.append(start_index)
                else:
                    break_index = 'Break'
                    ret1.append(break_index)
            except DoSCaBERError as e:
                e = 'None'
                ret1.append(e)
                pass

        ret2 = []
        for i in range(0,frame_number):
            try:
                if i>=start_f and i<break_f:
                    image = expt.get_nth_image(i)
                    width = expt.width_average_minimumroi_modified(image) 
                    ret2.append(width)
                elif i<start_f:
                    start_index = 'None'
                    ret2.append(start_index)
                else:
                    break_index = 'Break'
                    ret2.append(break_index)
            except DoSCaBERError as e:
                e = 'None'
                ret2.append(e)
                pass        

        ret3 = []
        for i in range(0,frame_number):
            try:
                if i>=start_f and i<break_f:
                    image = expt.get_nth_image(i)
                    width = expt.width_average_minimumroi_moving(image)
                    ret3.append(width)
                elif i<start_f:
                    start_index = 'None'
                    ret3.append(start_index)
                else:
                    break_index = 'Break'
                    ret3.append(break_index)
            except DoSCaBERError as e:
                e = 'None'
                ret3.append(e)
                pass

            
        time = []
        for i in range(1,len(ret)+1):
            t = i /fps
            time.append(t)
        data = {'time':time, 'Start':ret_s, 'Break':ret, 'width(fixed)':ret1, 'width(min)':ret2, 'widht(min_mov)' :ret3}
        data = pd.DataFrame(data)
        data.to_excel(excel_writer= directory+savenumber+'/'+'{}.xlsx'.format(savenumber),sheet_name='Radius Evolution')


        ## 해당부분을 다른 함수(모듈)로 구성하는게 좋을듯. 엑셀에서 데이터 불러오도록 하기.



    def Image_Radius_Measure(self,tifname,savenumber,fps):
        """
        측정영상(tif file)을 Frame 별로 분석하여 측정시간/ Break시간/ Wdith 값을 엑셀로 추출합니다.
        fps에 따라 code 정보 변경해야 합니다. (기본 10000 fps로 설정)

        Parameters
        ==========

        tifname : str
            측정이미지의 경로로, "filename.tif (or .tiff)" 형태로 입력합니다.

        savenumber : int
            data가 저장될 엑셀의 파일명으로, 20210723 과 같이 숫자를 입력합니다. (추후 수정 계획)

        Returns
        =======
        없음
        현재 경로에 data_savenumber.xlsx 파일을 형성합니다.

        Examples
        ========

        DoS CaBER 측정이미지를 통해, 시간(frame)에 따른 Neck의 Width 및 Break여부를 엑셀data로 추출합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path

        >>> expt.Dos_CaBER_fixed_min("sample_250fps.tiff",20210723) # doctest: +SKIP
        data_20210723.xlsx 파일 형성 # doctest: +SKIP

        Raises
        ======

        """
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from mcplexpt.caber.dos import DoSCaBERExperiment
        from mcplexpt.testing import get_samples_path
        path = get_samples_path("caber", "dos", tifname)
        expt = DoSCaBERExperiment(path)

        directory = os.getcwd()+'/relaxationtime/'
        try:
            if not os.path.exists(directory+savenumber):
                os.makedirs(directory+savenumber)
        except OSError:
            print ('Error: Creating directory. ' +  directory+savenumber)

        ret0=[]
        for i in range(0,10000000):
            try:
                image = expt.get_nth_image(i)
                ret0.append(i)
            except IndexError:
                break
        frame_number = len(ret0)

        ret_s = []
        for i in range(0,frame_number):
            image = expt.get_nth_image(i)
            result = expt.capbridge_start(image)
            ret_s.append(result)
        start_f = ret_s.index(True)
        ret = []
        for i in range(0,frame_number):
            image = expt.get_nth_image(i)
            result = expt.capbridge_broken(image)
            ret.append(result)
        break_f = ret.index(True)

        ##이미지 저장
        '''if break_f>20 and frame_number-break_f> 10:
            for i in range(break_f-20,break_f+10):
                image = expt.get_nth_image(i)
                cv2.imwrite(directory+savenumber+'/'+savenumber+'_image{}.png'.format(i),image)
        else: 
            for i in range(break_f-10,break_f+5):
                image = expt.get_nth_image(i)
                cv2.imwrite(directory+savenumber+'/'+savenumber+'_image{}.png'.format(i),image)'''
        
        ret1 = []
        for i in range(0,frame_number):
            try:
                if i>=start_f and i<break_f:
                    image = expt.get_nth_image(i)
                    width = expt.width_average_fixedroi_modified(image) 
                    ret1.append(width)
                elif i<start_f:
                    start_index = 'None'
                    ret1.append(start_index)
                else:
                    break_index = 'Break'
                    ret1.append(break_index)
            except DoSCaBERError as e:
                e = 'None'
                ret1.append(e)
                pass

        ret2 = []
        for i in range(0,frame_number):
            try:
                if i>=start_f and i<break_f:
                    image = expt.get_nth_image(i)
                    width = expt.width_average_minimumroi_modified(image) 
                    ret2.append(width)
                elif i<start_f:
                    start_index = 'None'
                    ret2.append(start_index)
                else:
                    break_index = 'Break'
                    ret2.append(break_index)
            except DoSCaBERError as e:
                e = 'None'
                ret2.append(e)
                pass        

        ret3 = []
        for i in range(0,frame_number):
            try:
                if i>=start_f and i<break_f:
                    image = expt.get_nth_image(i)
                    width = expt.width_average_minimumroi_moving(image)
                    ret3.append(width)
                elif i<start_f:
                    start_index = 'None'
                    ret3.append(start_index)
                else:
                    break_index = 'Break'
                    ret3.append(break_index)
            except DoSCaBERError as e:
                e = 'None'
                ret3.append(e)
                pass

            
        time = []
        for i in range(1,len(ret)+1):
            t = i /fps
            time.append(t)
        data = {'time':time, 'Start':ret_s, 'Break':ret, 'width(fixed)':ret1, 'width(min)':ret2, 'widht(min_mov)' :ret3}
        data = pd.DataFrame(data)
        data.to_excel(excel_writer= directory+savenumber+'/'+'{}.xlsx'.format(savenumber),sheet_name='Radius Evolution')


        ## 해당부분을 다른 함수(모듈)로 구성하는게 좋을듯. 엑셀에서 데이터 불러오도록 하기.

    def Dos_CaBER_VE_analysis(self,tifname,FE_time,fps):
        """
        Viscoelastic 물질 ( ex. 0.5 PEO solution)을 분석하여 Relaxation 값과 Extensional viscosity를 구함

        Parameters
        ==========

        tifname : str
            측정Data경로로, 'filename.xlsx' 형태로 입력합니다.

        Returns
        =======

        Examples
        ========

        Raises
        ======

        """
        import scipy.optimize as optimize
        import os
        import pandas as pd       
        import matplotlib.pyplot as plt 

        directory_1 = os.getcwd()+'/relaxationtime/'
        directory_2 = directory_1+tifname[:-5]
                     
        image_check = self.get_nth_image(-1)
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 :
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))

        Measurement_data = pd.read_excel(directory_2+'/'+'{}'.format(tifname),header=0,index_col=0,engine='openpyxl')
        time = Measurement_data['time'].values.tolist()
        ret1 = Measurement_data['width(fixed)'].values.tolist()
        ret2 = Measurement_data['width(min)'].values.tolist()
        ret3 = Measurement_data['widht(min_mov)'].values.tolist()

        ret_s =  np.array(ret3)
        index_s = np.where(ret_s == 'None')
        index_ss = list(index_s[0])
        if len(index_ss) == 0:
            index_start = 0
        else:
            index_start = index_ss[-1]        
        ret_b = np.array(ret3)
        index_b = np.where(ret_b == 'Break')
        index_bb = list(index_b[0])
        index_break = index_bb[0]

        time_slice = time[index_start+1:index_break]
        time_slice_shift = []
        for i in time_slice:
            result = i - time_slice[0]
            time_slice_shift.append(result)
        ret1_slice = ret1[index_start+1:index_break]
        ret2_slice = ret2[index_start+1:index_break]
        ret3_slice = ret3[index_start+1:index_break]

        xdata = np.array(time_slice_shift)
        try:
            global y1data
            y1data = (np.array(ret1_slice)) / nozzle_thick

        except TypeError:
            y1data =np.array(0)
            pass
            
        y2data = (np.array(ret2_slice)) / nozzle_thick
        y3data = (np.array(ret3_slice)) / nozzle_thick

        if np.size(y1data) != np.size(y2data):
            y1data = y2data

        y1data_log = np.log(y1data)
        y2data_log = np.log(y2data)
        y3data_log = np.log(y3data)

        '''OOP plot'''
        
        fig, axes= plt.subplots(nrows=2,ncols=2,figsize=(14,10))        
        ax1= axes[0,0]
        ax2= axes[0,1]
        ax3= axes[1,0]

        '''
        EC 영역 구분하기
        '''
        #이동평균(Moving Average)
        Y1data= []
        for y in range(len(y3data)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + y3data[y-f]
                Y1data.append(Y/30)
            elif y ==0:
                Y1data.append(y3data[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + y3data[y-f]
                Y1data.append(Y/len(range(0,y)))
        Y1data_n = np.array(Y1data)
        
        ax1.plot(xdata,Y1data_n,"bo")
        ax1.set_title('min_sort_data_raw')
        ax1.set_yscale('Log')
        ax1.set_ylabel('R/R0(mm)')
        ax1.set_xlabel('Time(s)')        


        #기울기 뽑기

        ret_slope = []
        for i in range(len(xdata)):
            if i > 9:
                slope = (Y1data[i]-Y1data[i-10]) / (xdata[i]-xdata[i-10])
                ret_slope.append(slope)
            else:
                slope = (Y1data[10]-Y1data[0])/(xdata[10]-xdata[0])
                ret_slope.append(slope)            
                
        Y2data= []
        for y in range(len(ret_slope)):                            
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope[y-f]
                Y2data.append(Y/30)
            elif y ==0:
                Y2data.append(ret_slope[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope[y-f]
                Y2data.append(Y/len(range(0,y)))
        Y2data_n = np.array(Y2data)

        ax2.plot(xdata,Y2data_n,"bo")
        ax2.set_title('min_sort_data_diff')
        ax2.set_ylabel('R/R0')
        ax2.set_xlabel('time')        


        print('Y2data',Y2data)

        #2차 미분값 뽑기                
     
        ret_slope2 = []

        for i in range(len(xdata)):
            if i > 9:
                slope2 = (Y2data[i]-Y2data[i-10]) / (xdata[i]-xdata[i-10])/fps
                ret_slope2.append(slope2)
            else:
                slope2 = (Y2data [10]-Y2data[0])/(xdata[10]-xdata[0])/fps
                ret_slope2.append(slope2)

        Y3data= []           
        
        for y in range(len(ret_slope2)):            
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope2[y-f]
                Y3data.append(Y/30)
            elif y ==0:
                Y3data.append(ret_slope2[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope2[y-f]
                Y3data.append(Y/len(range(0,y)))

        Y3data_n = np.array(Y3data)
        ax3.plot(xdata,Y3data_n,"bo")
        ax3.set_title('min_sort_data_diff_diff')
        ax3.set_ylabel('R/R0')
        ax3.set_xlabel('time')      

        print("Y3data",Y3data)

        ## log(y)의 선형구간(기울기 변화 없는 구간) 선정
        '''
        Y2data_filter = []
        xdata_filter = []

        for j in range(len(xdata)) :
            if Y2data[j] > minslope:
                xdata_filter.append(j)

        for i in Y2data:
            if i > minslope:
                Y2data_filter.append(i)

        print("Y2data_filter",Y2data_filter)
        print("xdata_filter",xdata_filter)

        ret_slope3 = []
        for i in range(len(xdata_filter)):
            if i > 9:
                slope3 = (Y2data_filter[i]-Y2data_filter[i-10]) / (xdata_filter[i]-xdata_filter[i-10])
                ret_slope3.append(slope3)
            else:
                slope3 = (Y2data_filter[10]-Y2data_filter[0])/(xdata_filter[10]-xdata_filter[0])
                ret_slope3.append(slope3)
    

        Y3data_filter= []           
        
        for y in range(len(ret_slope3)):            
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope3[y-f]
                Y3data_filter.append(Y/30)
            elif y ==0:
                Y3data_filter.append(ret_slope3[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope3[y-f]
                Y3data_filter.append(Y/len(range(0,y)))

        print("Y3data_filter",Y3data_filter)
        Y3data_n_abs = []
             
        try:
            for i in Y3data_filter:
                result = float(abs(i))
                if result <0.008 : 
                    result = 0                
                Y3data_n_abs.append(result)               
        except OverflowError:
            check = len(Y3data_filter) - len(Y3data_n_abs)
            while check != 0:
                check = check -1
                Y3data_n_abs.append(10)

        lenth = []
        lenth_ret = []
        for i in range(0,len(Y3data_n_abs)):
            if Y3data_n_abs[i] == 0 and i !=len(Y3data_n_abs)-1 :
                lenth_ret.append(0)
            else:
                if len(lenth_ret) ==0:
                    lenth.append(1)
                else:
                    lenth.append(len(lenth_ret)+1)
                    if len(lenth_ret) != 0 :
                        while len(lenth_ret) !=0:
                            lenth_ret.remove(0)
        print('Y3data_n_abs',Y3data_n_abs)
                        
        lenth_n = np.array(lenth)        
        index_lenth = np.where(lenth_n == max(lenth_n[10:])) ## 그려지는 그래프에 따라 VC구간이 Linear하게 계산될 수 있음.
        index_lenth_real = index_lenth[0][0]

        print('lenth_n',lenth_n)
        print('index_lenth',index_lenth)
        print('index_lenth_real',index_lenth_real)  
        '''
        
        Y2datanp=np.array(Y2data)
        IC_time =np.argmin(Y2datanp)
        print(IC_time)
        ict = xdata[0:IC_time]       
        icy = y1data[0:IC_time]
        
        
        icmodel = lambda ict,x,tc : np.log(x)+0.667*np.log((tc-ict)/0.00006)
        popt4, pcov4= optimize.curve_fit(icmodel, ict, np.log(icy))
        print(int(popt4[1]*fps))
        '''
        icpt_lenth=int(popt4[1]*10000)
        icpt=[]
        for i in range(icpt_lenth):
            icpt.append(i/10000)
        '''            
        EC_region_xdata = xdata[int(popt4[1]*fps):int(FE_time*fps/10000)]
        EC_region_y1data = y1data[int(popt4[1]*fps):int(FE_time*fps/10000)] 
        EC_region_y1data_log = y1data_log[int(popt4[1]*fps):int(FE_time*fps/10000)] 
        EC_region_y2data = y2data[int(popt4[1]*fps):int(FE_time*fps/10000)] 
        EC_region_y2data_log = y2data_log[int(popt4[1]*fps):int(FE_time*fps/10000)]
        EC_region_y3data = y3data[int(popt4[1]*fps):int(FE_time*fps/10000)] 
        EC_region_y3data_log = y3data_log[int(popt4[1]*fps):int(FE_time*fps/10000)]
    
        expt=xdata[int(popt4[1]*fps):int(FE_time*fps/10000)]
        expy1= EC_region_y1data
        expy2= EC_region_y2data
        expy3= EC_region_y3data
        print('expt',expt)


        model1 = lambda expt, a,b,c : -(expt-c)/3/b+np.log(a**(1/3))
        model2 = lambda expt, a,b,c : -(expt-c)/3/b+np.log(a**(1/3))
        model3 = lambda expt, a,b,c : -(expt-c)/3/b+np.log(a**(1/3))
        popt1, pcov1= optimize.curve_fit(model1, expt, np.log(expy1)) 
        popt2, pcov2= optimize.curve_fit(model2, expt, np.log(expy2)) 
        popt3, pcov3= optimize.curve_fit(model3, expt, np.log(expy3)) 
        fig, ax1 = plt.subplots(figsize=(6,6), dpi=80)
        

        ax1.plot(expt, np.exp(model1(expt,*popt1)),"ro", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt1))        
        ax1.semilogy(xdata, y1data, "b-")        
        plt.legend(fontsize=13) 
        plt.title('fixed_measure')
        fig.savefig('{}/{}'.format(directory_2,"{}_VE_Analysis1".format(tifname[:-5])))
        
        fig, ax2 = plt.subplots(figsize=(6,6), dpi=80)
        ax2.plot(expt, np.exp(model2(expt,*popt2)),"ro", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt2))
        ax2.semilogy(xdata, y2data, "b-")      
        plt.legend(fontsize=13) 
        plt.title('min_measure')        
        fig.savefig('{}/{}'.format(directory_2,"{}_VE_Analysis2".format(tifname[:-5])))

        fig, ax3 = plt.subplots(figsize=(6,6), dpi=80)
        ax3.plot(expt, np.exp(model3(expt,*popt3)),"ro", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt3))  
        ax3.semilogy(xdata, y3data, "b-")
             
        plt.legend(fontsize=13) 
        plt.title('min_sort_measure')
        fig.savefig('{}/{}'.format(directory_2,"{}_VE_Analysis3".format(tifname[:-5])))



    #Inertio Capillary fitting

        fig, ax3= plt.subplots(figsize=(6,6), dpi=80)

        ax3.plot(ict,np.exp(icmodel(ict,*popt4)),"ro", label="IC: X={:5.4f},tc={:5.4f}".format(*popt4))
        ax3.semilogy(xdata, y1data, "b-") 
        plt.legend(fontsize=13)
        fig.savefig('{}/{}'.format(directory_2,"{}_IC_Analysis".format(tifname[:-5])))

        npict= np.array(ict)
        npfit= np.array(np.exp(icmodel(ict,*popt4)))

        trict=np.transpose(npict)
        trfit=np.transpose(npfit)

        data1 = {'time':trict, 'fit' :trfit}
        data1 = pd.DataFrame(data1)
        data1.to_excel(excel_writer= directory_2+'/'+'IC.xlsx',sheet_name='Radius Evolution')

        print(trict,trfit) 


        

    #Inertio Elasto Plotting
        ax3.plot(expt, np.exp(model1(expt,*popt1)),"yo", label="EC: a={:5.4f}, λ={:5.4f}".format(*popt1))        
        plt.legend(fontsize=13) 
        plt.title('Inertio Elasto')
        ax3.set_ylabel("R/R0(mm)")
        ax3.set_xlabel("Time(s)")
        fig.savefig('{}/{}'.format(directory_2,"{}_IE_Analysis1".format(tifname[:-5])))


        
    #Extensional viscosity
    
        '''
        ret1_array_unitcontr = y1data*nozzle_thick*4/1000 # 4는 1픽셀 = 4um 의미함. 렌즈 배율변경시 변경필요.
        ret2_array_unitcontr = y2data*nozzle_thick*4/1000
        ret3_array_unitcontr = y3data*nozzle_thick*4/1000
        
        ret1_array_unitcontro = []
        for y in range(len(ret1_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret1_array_unitcontr[y-f]
                ret1_array_unitcontro.append(Y/30)
            elif y == 0:
                ret1_array_unitcontro.append(ret1_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret1_array_unitcontr[y-f]
                ret1_array_unitcontro.append(Y/len(range(0,y)))
        ret1_array_unitcontrol = np.array(ret1_array_unitcontro)
        
        ret2_array_unitcontro = []
        for y in range(len(ret2_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret2_array_unitcontr[y-f]
                ret2_array_unitcontro.append(Y/30)
            elif y == 0:
                ret2_array_unitcontro.append(ret2_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret2_array_unitcontr[y-f]
                ret2_array_unitcontro.append(Y/len(range(0,y)))
        ret2_array_unitcontrol = np.array(ret2_array_unitcontro)
        
        ret3_array_unitcontro = []
        for y in range(len(ret3_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret3_array_unitcontr[y-f]
                ret3_array_unitcontro.append(Y/30)
            elif y == 0:
                ret3_array_unitcontro.append(ret3_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret3_array_unitcontr[y-f]
                ret3_array_unitcontro.append(Y/len(range(0,y)))
        ret3_array_unitcontrol = np.array(ret3_array_unitcontro)
        
        term = 100

        ret1_radius_slope = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = (ret1_array_unitcontrol[i]-ret1_array_unitcontrol[i-term]) / (xdata[i]-xdata[i-term])
                ret1_radius_slope.append(slope)
            else:
                slope = (ret1_array_unitcontrol[term]-ret1_array_unitcontrol[0])/(xdata[term]-xdata[0])
                ret1_radius_slope.append(slope)
        ret2_radius_slope = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = (ret2_array_unitcontrol[i]-ret2_array_unitcontrol[i-term]) / (xdata[i]-xdata[i-term])
                ret2_radius_slope.append(slope)
            else:
                slope = (ret2_array_unitcontrol[term]-ret2_array_unitcontrol[0])/(xdata[term]-xdata[0])
                ret2_radius_slope.append(slope)
        ret3_radius_slope = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = (ret3_array_unitcontrol[i]-ret3_array_unitcontrol[i-term]) / (xdata[i]-xdata[i-term])
                ret3_radius_slope.append(slope)
            else:
                slope = (ret3_array_unitcontrol[term]-ret3_array_unitcontrol[0])/(xdata[term]-xdata[0])
                ret3_radius_slope.append(slope)

        surface_tension = surfacet   #측정물질에따라 변경필요.(jason이랑 연결하기)
        ret1_extensional_v = -surface_tension / np.array(ret1_radius_slope)
        ret2_extensional_v = -surface_tension / np.array(ret2_radius_slope)
        ret3_extensional_v = -surface_tension / np.array(ret3_radius_slope)
        
        ret1_extensional_v_slice = ret1_extensional_v[IC_time:FE_time] 
        ret2_extensional_v_slice = ret2_extensional_v[IC_time:FE_time] 
        ret3_extensional_v_slice = ret3_extensional_v[IC_time:FE_time] 
        
        hencky_strain1 = 2*np.log(1/y1data)
        hencky_strain2 = 2*np.log(1/y2data)
        hencky_strain3 = 2*np.log(1/y3data)
        
        hencky_strain1_slice = hencky_strain1[IC_time:FE_time]
        hencky_strain2_slice = hencky_strain2[IC_time:FE_time]
        hencky_strain3_slice = hencky_strain3[IC_time:FE_time]


        extensional_viscosity_1 = int(np.average(ret1_extensional_v_slice[-20:]))
        extensional_viscosity_2 = int(np.average(ret2_extensional_v_slice[-20:]))
        extensional_viscosity_3 = int(np.average(ret3_extensional_v_slice[-20:]))
        
        fig,ax4= plt.subplots(figsize=(5,5),dpi=100)
        fig,ax5= plt.subplots(figsize=(5,5),dpi=100)
        fig,ax6= plt.subplots(figsize=(5,5),dpi=100)
        
        ax4.plot(hencky_strain1_slice,ret1_extensional_v_slice,"ro",label = "Ex_viscosty = {}".format(extensional_viscosity_1))
        ax4.legend(fontsize=12) 
        ax4.set_title("fixed_roi_extensional viscoisty")
        ax4.set_yscale("log")
        ax4.set_yticks(np.logspace(0,4,5))
        ax4.set_ylabel("ηE")
        ax4.set_xlabel("ε")
        
        ax5.plot(hencky_strain2_slice,ret2_extensional_v_slice,"ro",label = "Ex_viscosty = {}".format(extensional_viscosity_2))
        ax5.legend(fontsize=12) 
        ax5.set_title("min_extensional viscoisity")
        ax5.set_yscale("log")
        ax5.set_yticks(np.logspace(0,4,5))
        ax5.set_ylabel("ηE")
        ax5.set_xlabel("ε")
        
        ax6.plot(hencky_strain3_slice,ret3_extensional_v_slice,"ro",label = "EC_time = {:5.5f}, Ex_viscosty = {}".format(tf, extensional_viscosity_3))
        ax6.legend(fontsize=12) 
        ax6.set_title("min_sort_extentsional viscoisty")
        ax6.set_yscale("log")
        ax6.set_yticks(np.logspace(0,4,5))
        ax6.set_ylabel("ηE")
        ax6.set_xlabel("ε")

        print(ret3_extensional_v_slice)
        '''
    def relaxationtime(self,fname1,fname2):
        import pandas as pd
        import os
        import matplotlib.pyplot as plt

        directory = os.getcwd()+'/'+'excelist'
        directory_2 = '/home/minhyukim/plotlist'
        Measurement_data1 = pd.read_excel('{}/{}'.format(directory,fname1),header=0,index_col=0,engine='openpyxl')
        Measurement_data2 =pd.read_excel('{}/{}'.format(directory,fname2),header=0,index_col=0,engine='openpyxl')
        ectime = Measurement_data1['time'].values.tolist()
        expy1 = Measurement_data1['radius'].values.tolist()
        expt = np.array(ectime)
        expy = np.array(expy1)

        model1 = lambda expt, a,b,c : -(expt-c)/3/b+np.log(a**(1/3))
        popt1, pcov1= optimize.curve_fit(model1, expt, np.log(expy)) 
        fig, ax1 = plt.subplots(figsize=(6,6), dpi=80)  

        ax1.plot(expt, np.exp(model1(expt,*popt1)),"ro", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt1))
        xdata = Measurement_data2['time']
        ydata = Measurement_data2['radius']

        ax1.semilogy(xdata, ydata, "b-")
        plt.legend(fontsize=13) 
        plt.title("{}".format(fname1[:-5]))
        fig.savefig('{}/{}'.format(directory_2,"{}_Relaxationtime".format(fname1[:-5])))

    def viscofitting(self,fname1,surfacet,zsv):
        import pandas as pd
        import os
        import matplotlib.pyplot as plt

        directory = os.getcwd()+'/'+'excelist'
        directory_2 = '/home/minhyukim/plotlist'
        Measurement_data1 = pd.read_excel('{}/{}'.format(directory,fname1),header=0,index_col=0,engine='openpyxl')
        vctime = Measurement_data1['time'].values.tolist()
        expy1 = Measurement_data1['radius'].values.tolist()
        expt = np.array(vctime)
        expy = np.array(expy1)

        model1 = lambda expt,c : 0.0709*surfacet/zsv/0.635*(c-expt)
        popt1, pcov1= optimize.curve_fit(model1, expt, np.log(expy)) 
        fig, ax1 = plt.subplots(figsize=(6,6), dpi=80)  

        ax1.plot(expt, np.exp(model1(expt,*popt1)),"ro", label="fit: c={:5.4f}".format(*popt1))
        xdata = Measurement_data2['time']
        ydata = Measurement_data2['radius']

        ax1.semilogy(xdata, ydata, "b-")
        plt.legend(fontsize=13) 
        plt.title("{}".format(fname1[:-5]))
        fig.savefig('{}/{}'.format(directory_2,"{}_viscoanalysis".format(fname1[:-5])))

    def powerlawfitting(self,fname1,percent,inv1,inv2,inv3):
        import pandas as pd
        import os
        import matplotlib.pyplot as plt

        directory = os.getcwd()+'/relaxationtime/'+fname1[:-5]
        Measurement_data1 = pd.read_excel('{}/{}'.format(directory,fname1),header=0,index_col=0,engine='openpyxl')
        vctime = Measurement_data1['time'].values.tolist()
        expy1 = Measurement_data1['width(min)'].values.tolist()
        totalt = np.array(vctime)
        print(len(vctime))
        fittingt=totalt[(-int(percent*len(vctime))):]
        print(fittingt)
        totaly = np.array(expy1)
        if 'None' in totaly :
            ntotaly=np.delete(totaly,np.where((totaly=='None')))
            ntotaly=np.array(ntotaly,dtype=np.float64)
            nortotaly=ntotaly/np.max(ntotaly)
        else:            
            nortotaly=totaly/np.max(totaly)
        fittingy=nortotaly[(-int(percent*len(vctime))):]
        print(fittingy)

        model1 = lambda fittingt,c,n,a : n*np.log(c-fittingt) + a +np.log(0.071+0.239*(1-n)+0.548*(1-n)**2)
        popt1, pcov1= optimize.curve_fit(model1, fittingt,np.log(fittingy),maxfev=10000,p0=[inv1,inv2,inv3]) 
        fig, ax1 = plt.subplots(figsize=(6,6), dpi=80)
        residuals = np.log(fittingy)- model1(fittingt, *popt1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.log(fittingy)-np.mean(np.log(fittingy)))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(r_squared)
        ax1.plot(fittingt, np.exp(model1(fittingt,*popt1)),"ro", label="fit: tc={:5.4f},n={:5.4f},σ/K={:5.4f}".format(*popt1))
        xdata = totalt[-len(nortotaly):]
        ydata = nortotaly

        ax1.semilogy(xdata, ydata, "b-")
        ax1.set_xlabel('tc-time')
        ax1.set_ylabel('R/R0')
        plt.legend(fontsize=13) 
        plt.title("{}".format(fname1[:-5]))
        fig.savefig('{}/{}'.format(directory,"{}_powerlawanalysis".format(fname1[:-5])))



    def TimeStretchingplot(self,fname,sparameter):
        import pandas as pd
        import os
        import matplotlib.pyplot as plt

        directory = os.getcwd()+'/'+'excelist'
        Measurement_data = pd.read_excel('{}/{}'.format(directory,fname),header=0,index_col=0,engine='openpyxl')
        time = Measurement_data['time'].values.tolist()
        ret1 = Measurement_data['width(fixed)'].values.tolist()
        ret2 = Measurement_data['width(min)'].values.tolist()
        ret3 = Measurement_data['widht(min_mov)'].values.tolist()



        ret_s =  np.array(ret3)
        index_s = np.where(ret_s == 'None')
        index_ss = list(index_s[0])

        if len(index_ss) == 0:
         index_start = 0
        else:
         index_start = index_ss[-1]

        ret_b = np.array(ret3)
        index_b = np.where(ret_b == 'Break')
        index_bb = list(index_b[0])
        index_break = index_bb[0]

        time_slice = time[index_start+1:index_break]
        time_slice_shift = []
        for i in time_slice:
            result = i - time_slice[0]
            time_slice_shift.append(result)

        stt=[]

        for t in time_slice_shift:
            st= sparameter*t
            stt.append(st)
        
        
        ret3_slice = ret3[index_start+1:index_break]
        
        image_check = self.get_nth_image(-1)
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        row_l=[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 and thick < int(sum(thick_l)/len(thick_l)-10) : ## 노즐 두께에 따라 310이 아닐수 있기때문에 변경필요.
                row_l.append(row)
                break    
        nozzle_thick = int(sum(thick_l)/len(thick_l))
   
        y3data = (np.array(ret3_slice)) / nozzle_thick*4/1000

        data = {'time':stt, 'Radius':y3data}
        data = pd.DataFrame(data)
              
        data.to_excel(excel_writer= directory+'/'+'stretchedexcellist'+'/'+'{}'.format(str(sparameter))+'_'+str(fname)+'.xlsx',sheet_name='stretching')


        fig1,ax1= plt.subplots(figsize=(7,7),dpi=100)
        ax1.set_ylabel("Radius(mm)",fontsize=13)
        ax1.set_yscale("log")
        ax1.set_xlabel("Stretched time(s)",fontsize=13)
        ax1.plot(stt,y3data,"ro",label='{}'.format(sparameter))
        ax1.legend(fontsize=12)
        plt.legend(loc='upper center')
        fig1.savefig('/home/minhyukim/stretchedtimeplot/'+'{}'.format(fname[:6]))

    def powerlawplot(self,fname):

        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import scipy.optimize as optimize


        #################################

        directory_2 = os.getcwd()+'/relaxationtime/'+fname[:-5]
        image_check = self.get_nth_image(-1)
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 :
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))

        Measurement_data = pd.read_excel('{}/{}'.format(directory_2,fname),header=0,index_col=0,engine='openpyxl')
        time = Measurement_data['time'].values.tolist()
        ret1 = Measurement_data['width(fixed)'].values.tolist()
        
        ret_s =  np.array(ret1)
        index_s = np.where(ret_s == 'None')
        index_ss = list(index_s[0])
        if len(index_ss) == 0:
            index_start = 0
        else:
            index_start = index_ss[-1]

        ret_b = np.array(ret1)
        index_b = np.where(ret_b == 'Break')
        index_bb = list(index_b[0])
        index_break = len(ret1)

        time_slice = time[index_start+1:index_break]
        time_slice_shift = []
        for i in time_slice:
            result = i - time_slice[0]
            time_slice_shift.append(result)
        ret1_slice = ret1[index_start+1:index_break]

        xdata = np.array(time_slice_shift)
        try:
            global y1data
            y1data = (np.array(ret1_slice)) / nozzle_thick

        except TypeError:
            y1data =np.array(0)
            pass            
        
            
        ret1_array_unitcontr = y1data*nozzle_thick*4/1000 # 4는 1픽셀 = 4um 의미함. 렌즈 배율변경시 변경필요.
        
        ## 30개 term 으로  moving average 적용함. (초기 30개 data는 해당구간까지만 term 계산하여 각각 moving average 적용)
        ret1_array_unitcontro = []
        for y in range(len(ret1_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret1_array_unitcontr[y-f]
                ret1_array_unitcontro.append(Y/30)
            elif y == 0:
                ret1_array_unitcontro.append(ret1_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret1_array_unitcontr[y-f]
                ret1_array_unitcontro.append(Y/len(range(0,y)))
        ret1_array_unitcontrol = np.array(ret1_array_unitcontro)
        
        ret2_array_unitcontro = []

        #############################################################
        '''OOP plot'''
        
        fig, axes= plt.subplots(nrows=2,ncols=2,figsize=(14,10))        
        ax1= axes[0,0]
        ax2= axes[0,1]
        ax3= axes[1,0]
    

        '''
        EC 영역 구분하기
        '''
        #이동평균(Moving Average)
        Y1data= []
        for y in range(len(y1data)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + y1data[y-f]
                Y1data.append(Y/30)
            elif y ==0:
                Y1data.append(y1data[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + y1data[y-f]
                Y1data.append(Y/len(range(0,y)))
        Y1data_n = np.array(Y1data)

        ax1.plot(xdata,Y1data_n,"bo")
        ax1.set_title('min_sort_data_raw',fontsize=18)
        ax1.set_yscale('Log')
        ax1.set_ylabel('R/R0',fontsize=18)
        ax1.set_xlabel('time',fontsize=18)        


        #기울기 뽑기

        ret_slope = []
        for i in range(len(xdata)):
            if i > 9:
                slope = (Y1data[i]-Y1data[i-10]) / (xdata[i]-xdata[i-10])
                ret_slope.append(slope)
            else:
                slope = (Y1data[10]-Y1data[0])/(xdata[10]-xdata[0])
                ret_slope.append(slope)
                
        Y2data= []
        for y in range(len(ret_slope)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope[y-f]
                Y2data.append(Y/30)
            elif y ==0:
                Y2data.append(ret_slope[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope[y-f]
                Y2data.append(Y/len(range(0,y)))
        Y2data_n = np.array(Y2data)

        ax2.plot(xdata,Y2data_n,"bo")
        ax2.set_title('min_sort_data_diff',fontsize=18)
        ax2.set_ylabel('R/R0',fontsize=18)
        ax2.set_xlabel('time',fontsize=18)        



        #2차 미분값 뽑기

        ret_slope2 = []
        for i in range(len(xdata)):
            if i > 9:
                slope = (Y2data[i]-Y2data[i-10]) / (xdata[i]-xdata[i-10])/10000
                ret_slope2.append(slope)
            else:
                slope = (Y2data[10]-Y2data[0])/(xdata[10]-xdata[0])/10000
                ret_slope2.append(slope)
                
        Y3data= []
        for y in range(len(ret_slope2)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope2[y-f]
                Y3data.append(Y/30)
            elif y ==0:
                Y3data.append(ret_slope2[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope2[y-f]
                Y3data.append(Y/len(range(0,y)))
        Y3data_n = np.array(Y3data)

        ax3.plot(xdata,Y3data_n,"bo")
        ax3.set_title('min_sort_data_diff_diff',fontsize=18)
        ax3.set_ylabel('R/R0',fontsize=18)
        ax3.set_xlabel('time',fontsize=18)      

        fig.savefig('{}/{}'.format(directory_2,"{}_Radius Evolution".format(fname[:-5])))
        






    def Extensional_viscosity_total(self,fname,surfacet):

        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import scipy.optimize as optimize


        #################################

        directory_2 = os.getcwd()+'/relaxationtime/'+fname[:-5]
        image_check = self.get_nth_image(-1)
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 :
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))

        Measurement_data = pd.read_excel('{}/{}'.format(directory_2,fname),header=0,index_col=0,engine='openpyxl')
        time = Measurement_data['time'].values.tolist()
        ret1 = Measurement_data['width(fixed)'].values.tolist()
        ret2 = Measurement_data['width(min)'].values.tolist()
        ret3 = Measurement_data['widht(min_mov)'].values.tolist()
        
        ret_s =  np.array(ret3)
        index_s = np.where(ret_s == 'None')
        index_ss = list(index_s[0])
        if len(index_ss) == 0:
            index_start = 0
        else:
            index_start = index_ss[-1]

        ret_b = np.array(ret3)
        index_b = np.where(ret_b == 'Break')
        index_bb = list(index_b[0])
        index_break = index_bb[0]

        time_slice = time[index_start+1:index_break]
        time_slice_shift = []
        for i in time_slice:
            result = i - time_slice[0]
            time_slice_shift.append(result)
        ret1_slice = ret1[index_start+1:index_break]
        ret2_slice = ret2[index_start+1:index_break]
        ret3_slice = ret3[index_start+1:index_break]

        xdata = np.array(time_slice_shift)
        try:
            global y1data
            y1data = (np.array(ret1_slice)) / nozzle_thick

        except TypeError:
            y1data =np.array(0)
            pass
            
        y2data = (np.array(ret2_slice)) / nozzle_thick
        y3data = (np.array(ret3_slice)) / nozzle_thick
        
        if np.size(y1data) != np.size(y2data):
            y1data = y2data
            
        ret1_array_unitcontr = y1data*nozzle_thick*4/1000 # 4는 1픽셀 = 4um 의미함. 렌즈 배율변경시 변경필요.
        ret2_array_unitcontr = y2data*nozzle_thick*4/1000
        ret3_array_unitcontr = y3data*nozzle_thick*4/1000
        
        ## 30개 term 으로  moving average 적용함. (초기 30개 data는 해당구간까지만 term 계산하여 각각 moving average 적용)
        ret1_array_unitcontro = []
        for y in range(len(ret1_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret1_array_unitcontr[y-f]
                ret1_array_unitcontro.append(Y/30)
            elif y == 0:
                ret1_array_unitcontro.append(ret1_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret1_array_unitcontr[y-f]
                ret1_array_unitcontro.append(Y/len(range(0,y)))
        ret1_array_unitcontrol = np.array(ret1_array_unitcontro)
        
        ret2_array_unitcontro = []
        for y in range(len(ret2_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret2_array_unitcontr[y-f]
                ret2_array_unitcontro.append(Y/30)
            elif y == 0:
                ret2_array_unitcontro.append(ret2_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret2_array_unitcontr[y-f]
                ret2_array_unitcontro.append(Y/len(range(0,y)))
        ret2_array_unitcontrol = np.array(ret2_array_unitcontro)
        
        ret3_array_unitcontro = []
        for y in range(len(ret3_array_unitcontr)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y+ret3_array_unitcontr[y-f]
                ret3_array_unitcontro.append(Y/30)
            elif y == 0:
                ret3_array_unitcontro.append(ret3_array_unitcontr[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret3_array_unitcontr[y-f]
                ret3_array_unitcontro.append(Y/len(range(0,y)))
        ret3_array_unitcontrol = np.array(ret3_array_unitcontro)

        #############################################################
        '''OOP plot'''
        
        fig, axes= plt.subplots(nrows=2,ncols=2,figsize=(14,10))        
        ax1= axes[0,0]
        ax2= axes[0,1]
        ax3= axes[1,0]
    

        '''
        EC 영역 구분하기
        '''
        #이동평균(Moving Average)
        Y1data= []
        for y in range(len(y3data)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + y3data[y-f]
                Y1data.append(Y/30)
            elif y ==0:
                Y1data.append(y3data[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + y3data[y-f]
                Y1data.append(Y/len(range(0,y)))
        Y1data_n = np.array(Y1data)

        ax1.plot(xdata,Y1data_n,"bo")
        ax1.set_title('min_sort_data_raw',fontsize=18)
        ax1.set_yscale('Log')
        ax1.set_ylabel('R/R0',fontsize=18)
        ax1.set_xlabel('time',fontsize=18)        


        #기울기 뽑기

        ret_slope = []
        for i in range(len(xdata)):
            if i > 9:
                slope = (Y1data[i]-Y1data[i-10]) / (xdata[i]-xdata[i-10])
                ret_slope.append(slope)
            else:
                slope = (Y1data[10]-Y1data[0])/(xdata[10]-xdata[0])
                ret_slope.append(slope)
                
        Y2data= []
        for y in range(len(ret_slope)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope[y-f]
                Y2data.append(Y/30)
            elif y ==0:
                Y2data.append(ret_slope[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope[y-f]
                Y2data.append(Y/len(range(0,y)))
        Y2data_n = np.array(Y2data)

        ax2.plot(xdata,Y2data_n,"bo")
        ax2.set_title('min_sort_data_diff',fontsize=18)
        ax2.set_ylabel('R/R0',fontsize=18)
        ax2.set_xlabel('time',fontsize=18)        



        #2차 미분값 뽑기

        ret_slope2 = []
        for i in range(len(xdata)):
            if i > 9:
                slope = (Y2data[i]-Y2data[i-10]) / (xdata[i]-xdata[i-10])/10000
                ret_slope2.append(slope)
            else:
                slope = (Y2data[10]-Y2data[0])/(xdata[10]-xdata[0])/10000
                ret_slope2.append(slope)
                
        Y3data= []
        for y in range(len(ret_slope2)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + ret_slope2[y-f]
                Y3data.append(Y/30)
            elif y ==0:
                Y3data.append(ret_slope2[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + ret_slope2[y-f]
                Y3data.append(Y/len(range(0,y)))
        Y3data_n = np.array(Y3data)

        ax3.plot(xdata,Y3data_n,"bo")
        ax3.set_title('min_sort_data_diff_diff',fontsize=18)
        ax3.set_ylabel('R/R0',fontsize=18)
        ax3.set_xlabel('time',fontsize=18)      

        fig.savefig('{}/{}'.format(directory_2,"{}_Radius Evolution".format(fname[:-5])))
        
        ##########################################
       ## if len(xdata)>=100:
       ##     term = 100
       ## else:
       ##     term = 10

        term = int(len(xdata)/10)

        ret1_radius_slope = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = (ret1_array_unitcontrol[i]-ret1_array_unitcontrol[i-term]) / (xdata[i]-xdata[i-term])
                ret1_radius_slope.append(slope)
            else:
                slope = (ret1_array_unitcontrol[term]-ret1_array_unitcontrol[0])/(xdata[term]-xdata[0])
                ret1_radius_slope.append(slope)
        ret2_radius_slope = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = (ret2_array_unitcontrol[i]-ret2_array_unitcontrol[i-term]) / (xdata[i]-xdata[i-term])
                ret2_radius_slope.append(slope)
            else:
                slope = (ret2_array_unitcontrol[term]-ret2_array_unitcontrol[0])/(xdata[term]-xdata[0])
                ret2_radius_slope.append(slope)
        ret3_radius_slope = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = (ret3_array_unitcontrol[i]-ret3_array_unitcontrol[i-term]) / (xdata[i]-xdata[i-term])
                ret3_radius_slope.append(slope)
            else:
                slope = (ret3_array_unitcontrol[term]-ret3_array_unitcontrol[0])/(xdata[term]-xdata[0])
                ret3_radius_slope.append(slope)

        ret1_strainrate = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = -2*(ret1_array_unitcontrol[i]-ret1_array_unitcontrol[i-term])/(ret1_array_unitcontrol[i]) / (xdata[i]-xdata[i-term])
                ret1_strainrate.append(slope)
            else:
                slope = -2*(ret1_array_unitcontrol[term]-ret1_array_unitcontrol[0])/(ret1_array_unitcontrol[term])/(xdata[term]-xdata[0])
                ret1_strainrate.append(slope)
        ret2_strainrate = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = -2*(ret2_array_unitcontrol[i]-ret2_array_unitcontrol[i-term])/(ret2_array_unitcontrol[i]) / (xdata[i]-xdata[i-term])
                ret2_strainrate.append(slope)
            else:
                slope = -2*(ret2_array_unitcontrol[term]-ret2_array_unitcontrol[0])/(ret2_array_unitcontrol[term])/(xdata[term]-xdata[0])
                ret2_strainrate.append(slope)
        ret3_strainrate = []
        for i in range(len(xdata)):
            if i > term-1:
                slope = -2*(ret3_array_unitcontrol[i]-ret3_array_unitcontrol[i-term])/(ret3_array_unitcontrol[i]) / (xdata[i]-xdata[i-term])
                ret3_strainrate.append(slope)
            else:
                slope = -2*(ret3_array_unitcontrol[term]-ret3_array_unitcontrol[0])/(ret3_array_unitcontrol[term])/(xdata[term]-xdata[0])
                ret3_strainrate.append(slope)

        surface_tension = surfacet  #측정물질에따라 변경필요.(jason이랑 연결하기)
        ret1_extensional_v = -surface_tension / np.array(ret1_radius_slope)
        ret2_extensional_v = -surface_tension / np.array(ret2_radius_slope)
        ret3_extensional_v = -surface_tension / np.array(ret3_radius_slope)
        
        print('ret1_extensional_v=',ret1_extensional_v , 'ret2_extensional_v=',ret2_extensional_v, 'ret3_extensional_v=',ret3_extensional_v)

        data = {'time':time_slice,'width(fixed)':ret1_array_unitcontro, 'width(min)':ret2_array_unitcontro, 'widht(min_mov)' :ret3_array_unitcontro, 'ne(fixed)':ret1_extensional_v,'ne(min)':ret2_extensional_v,'ne(min_sort)':ret3_extensional_v,'Strainrate(min_sort)':ret3_strainrate}
        data = pd.DataFrame(data)
        with pd.ExcelWriter('{}/{}'.format(directory_2,fname),engine='openpyxl',mode='a') as writer:
            data.to_excel(writer,sheet_name='Extensional V')
        
        hencky_strain1 = 2*np.log(1/y1data)
        hencky_strain2 = 2*np.log(1/y2data)
        hencky_strain3 = 2*np.log(1/y3data)
        
        fig,ax4= plt.subplots(figsize=(7,7),dpi=100)

        if max(ret1_extensional_v) > 100:
            ret1_extensional_v_maxy = 200
        else:
            ret1_extensional_v_maxy = max(ret1_extensional_v)

        if max(ret2_extensional_v) > 100:
            ret2_extensional_v_maxy = 200
        else:
            ret2_extensional_v_maxy = max(ret2_extensional_v)           

        if max(ret3_extensional_v) > 100:
            ret3_extensional_v_maxy = 200
        else:
            ret3_extensional_v_maxy = max(ret3_extensional_v)

        ax4.plot(ret1_slice,ret1_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax4.legend(fontsize=12) 
        ax4.set_ylim([0,ret1_extensional_v_maxy])
        ax4.set_title("{}fixed_roi_extensional viscoisty".format(fname[:-5]),fontsize=18)
        ax4.set_ylabel("ηE",fontsize=18)
        ax4.set_xlabel("R",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}fixed_roi_extensional viscoisty".format(fname[:-5])))

        fig,ax5= plt.subplots(figsize=(7,7),dpi=100)

        ax5.plot(ret2_slice,ret2_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax5.legend(fontsize=12) 
        ax5.set_ylim([0,ret2_extensional_v_maxy])
        ax5.set_title("{}min_extensional viscoisty".format(fname[:-5]),fontsize=18)
        ax5.set_ylabel("ηE",fontsize=18)
        ax5.set_xlabel("R",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}min_extensional viscoisty".format(fname[:-5])))
        
        fig,ax6= plt.subplots(figsize=(7,7),dpi=100)

        ax6.plot(ret3_slice,ret3_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax6.legend(fontsize=12) 
        ax6.set_ylim([0,ret3_extensional_v_maxy])
        ax6.set_title("{}min_sort_extensional viscoisty".format(fname[:-5]),fontsize=18)
        ax6.set_ylabel("ηE",fontsize=18)
        ax6.set_xlabel("R",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}min_sort_extensional viscoisty".format(fname[:-5])))

        fig,ax7= plt.subplots(figsize=(7,7),dpi=100)

        ax7.plot(ret1_strainrate,ret1_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax7.legend(fontsize=12) 
        ax7.set_ylim([0,ret1_extensional_v_maxy])
        ax7.set_title("{}fixed_roi_extensional viscoisty_sr".format(fname[:-5]),fontsize=18)
        ax7.set_ylabel("ηE",fontsize=18)
        ax7.set_xlabel("Strain Rate",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}fixed_roi_extensional viscoisty_sr".format(fname[:-5])))
        
        fig,ax8= plt.subplots(figsize=(7,7),dpi=100)
    
        ax8.plot(ret2_strainrate,ret2_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax8.legend(fontsize=12) 
        ax8.set_ylim([0,ret2_extensional_v_maxy])
        ax8.set_title("{}min_extensional viscoisty_sr".format(fname[:-5]),fontsize=18)
        ax8.set_ylabel("ηE",fontsize=18)
        ax8.set_xlabel("Strain Rate",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}min_extensional viscoisty_sr".format(fname[:-5])))
        
        fig,ax9= plt.subplots(figsize=(7,7),dpi=100)

        ax9.plot(ret3_strainrate,ret3_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax9.legend(fontsize=12) 
        ax9.set_ylim([0,ret3_extensional_v_maxy])
        ax9.set_title("{}min_sort_extensional viscoisty_sr".format(fname[:-5]),fontsize=18)
        ax9.set_ylabel("ηE",fontsize=18)
        ax9.set_xlabel("Strain Rate",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}min_sort_extensional viscoisty_sr".format(fname[:-5])))

        fig,ax99= plt.subplots(figsize=(7,7),dpi=100)

        ax99.plot(ret3_slice,ret3_strainrate,"ro",label='surf_Tension={}'.format(surfacet))
        ax99.legend(fontsize=12) 
        ax99.set_ylim([0,int(max(ret3_strainrate))])
        ax99.set_title("{}min_sort_strainrate_Radius".format(fname[:-5]),fontsize=18)
        ax99.set_ylabel("Strain Rate",fontsize=18)
        ax99.set_xlabel("R",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}min_sort_strainrate_Radius".format(fname[:-5])))
        

        fig,ax10= plt.subplots(figsize=(7,7),dpi=100)

        if max(ret1_extensional_v) > 100:
            ret1_extensional_v_maxy = 200
        else:
            ret1_extensional_v_maxy = max(ret1_extensional_v)

        if max(ret2_extensional_v) > 100:
            ret2_extensional_v_maxy = 200
        else:
            ret2_extensional_v_maxy = max(ret2_extensional_v)           

        if max(ret3_extensional_v) > 100:
            ret3_extensional_v_maxy = 200
        else:
            ret3_extensional_v_maxy = max(ret3_extensional_v)

        fig,ax10= plt.subplots(figsize=(7,7),dpi=100)
        ax10.plot(hencky_strain1,ret1_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax10.legend(fontsize=12) 
        ax10.set_ylim([0,ret1_extensional_v_maxy])
        ax10.set_title("{}fixed_roi_extensional viscoisty".format(fname[:-5]),fontsize=18)
        ax10.set_ylabel("ηE",fontsize=18)
        ax10.set_xlabel("hencky strain",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}_fixed_roi_extensional viscoisty".format(fname[:-5])))

        fig,ax11= plt.subplots(figsize=(7,7),dpi=100)
        ax11.plot(hencky_strain2,ret2_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax11.legend(fontsize=12) 
        ax11.set_ylim([0,ret2_extensional_v_maxy])
        ax11.set_title("{}min_extensional viscoisty".format(fname[:-5]),fontsize=18)
        ax11.set_ylabel("ηE",fontsize=18)
        ax11.set_xlabel("hencky strain",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}_min_extensional viscoisty".format(fname[:-5])))
        
        fig,ax12= plt.subplots(figsize=(7,7),dpi=100)
        ax12.plot(hencky_strain3,ret3_extensional_v,"ro",label='surf_Tension={}'.format(surfacet))
        ax12.legend(fontsize=12) 
        ax12.set_ylim([0,ret3_extensional_v_maxy])
        ax12.set_title("{}min_sort_extensional viscoisty".format(fname[:-5]),fontsize=18)
        ax12.set_ylabel("ηE",fontsize=18)
        ax12.set_xlabel("hencky strain",fontsize=18)
        fig.savefig('{}/{}'.format(directory_2,"{}_min_sort_extensional viscoisty".format(fname[:-5])))


        return ret1_extensional_v, ret2_extensional_v, ret3_extensional_v


    def Fast_Data_analysis(self,date,surfacet):
        """
        특정경로(C:/Users/MCPL-JJ/Desktop/codes/mcplexpt/samples/caber/dos)에 있는 특정 DoS-CaBER 측정 영상을 분석합니다. 

        Parameters
        ==========

        date : int(220212) or str(''형태)
            분석하고자 하는 영상을 특정합니다. 
            입력한 int 또는 str이 포함된 영상만을 분석합니다.

        surfacet : int
            해당 유체의 surfacetension을 입력합니다.

        Returns
        =======
        없음
        현재경로(os.getcwd())에 영상이름과 동일한 폴더를 만들고 내부에 분석결과가 저장됩니다.

        Examples
        ========

        Raises
        ======

        """
        import os
        path = "/home/minhyukim/samples/caber/dos"
        file_list = os.listdir(path)
        tif_list=[]
        for i in file_list:
            if 'tif' in i:
                tif_list.append(i)
        date_list=[]
        for i in tif_list:
            if '{}'.format(date) in i:
                date_list.append(i)
        savenumber = []
        for i in date_list:
            i = i[:-4]
            savenumber.append(i)
        excel_list = []
        for i in savenumber:
            i = i + '.xlsx'
            excel_list.append(i)

        for i in range(0,len(date_list)):
            self.Image_Radius_Measure(date_list[i],savenumber[i])
            self.Extensional_viscosity_total(excel_list[i],surfacet)






    def Image_storage(self,date,frame):
        """
        측정영상(tif file)을 Frame 별로 분석하여 측정시간/ Break시간/ Wdith 값을 엑셀로 추출합니다.
        fps에 따라 code 정보 변경해야 합니다. (기본 10000 fps로 설정)

        Parameters
        ==========

        tifname : str
            측정이미지의 경로로, "filename.tif (or .tiff)" 형태로 입력합니다.

        savenumber : int
            data가 저장될 엑셀의 파일명으로, 20210723 과 같이 숫자를 입력합니다. (추후 수정 계획)

        Returns
        =======
        없음
        현재 경로에 data_savenumber.xlsx 파일을 형성합니다.

        Examples
        ========

        DoS CaBER 측정이미지를 통해, 시간(frame)에 따른 Neck의 Width 및 Break여부를 엑셀data로 추출합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path

        >>> expt.Dos_CaBER_fixed_min("sample_250fps.tiff",20210723) # doctest: +SKIP
        data_20210723.xlsx 파일 형성 # doctest: +SKIP

        Raises
        ======

        """
        import os

        path1 = '/home/minhyukim/samples/caber/dos'
        file_list = os.listdir(path1)
        tif_list=[]
        for i in file_list:
            if '.tif' in i:
                tif_list.append(i)
        date_list=[]
        for i in tif_list:
            if '{}'.format(date) in i:
                date_list.append(i)
        savenumber = []
        for i in date_list:
            i = i[:-4]
            savenumber.append(i)
        for i in range(0,len(date_list)):
            print(savenumber[i])
            sys.stdout.flush()
            self.image_save(date_list[i],savenumber[i],frame)
   


    def image_save(self,fname,savenumber,frame):
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from mcplexpt.caber.dos import DoSCaBERExperiment
        from mcplexpt.testing import get_samples_path
        path = get_samples_path("caber", "dos", fname)
        expt = DoSCaBERExperiment(path)
        directory = os.getcwd()+'/pcalist'
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

        ret0=[]
        for i in range(0,10000000):
            try:
                image = expt.get_nth_image(i)
                ret0.append(i)
            except IndexError:
                break
        frame_number = len(ret0)

        ret_s = []
        for i in range(0,frame_number):
            image = expt.get_nth_image(i)
            result = expt.capbridge_start(image)
            ret_s.append(result)
        start_f = ret_s.index(True)
        
        ret = []
        for i in range(0,frame_number):
            image = expt.get_nth_image(i)
            result = expt.capbridge_broken(image)
            ret.append(result)
        break_f = ret.index(True)

        ret_t = []
        try:
            for i in range(0,frame_number):
                image = expt.get_nth_image(i)
                result = expt.capbridge_terminal(image)
                ret_t.append(result)
            break_t = ret_t.index(True)
        except ValueError:
            break_t = break_f+1
            pass

        if break_f == break_t:
            break_t = break_f + 1
        '''
        ##이미지 저장
        if break_f>20 and frame_number-break_f> 10:
            for i in range(break_f-20,break_f+10):
                image = expt.get_nth_image(i)
                cv2.imwrite(directory+savenumber+'/'+savenumber+'_image{}.png'.format(i),image)
        else: 
            for i in range(break_f-10,break_f+5):
                image = expt.get_nth_image(i)
                cv2.imwrite(directory+savenumber+'/'+savenumber+'_image{}.png'.format(i),image)
        '''
        ##이미지 저장(멀티 병합 이미지)
        image_list = []
        print(int(frame/100*(break_t-start_f)))
        for i in range(start_f,break_t):
            image = expt.get_nth_image(i)
            ##여기서 image crop해서 PEO용액 이상한 점 부분 없애고, 필요없는 부분 잘라내는게 좋아보임.
            image=image.download()
            image[514:517,88:91] = 180
            image_c = cv2.Canny(image,50,220)        
            yss,xss = np.where(image_c ==255)
            for y in set(yss):
                edge_xss = xss[np.where(yss==y)]
                for j in range(edge_xss[0],edge_xss[-1]):
                    image_c[y][j] = 255
            '''
            ys,xs=np.where(image_c> 250) ## cropped 범위 및 Center Align 수정
            xset=[]
            for i in set(ys):
                xi=xs[np.where(ys==i)]
                xc=(xi[0]+xi[-1])/2
                xset.append(xc)
            averagecenter=sum(xset)/len(set(ys))
            image_sc = image_c[140:500,int(averagecenter-200):int(averagecenter+200)]    
            '''
            '''       
            image_c = image_c/(break_f-start_f) ## 22.08.16 일단 Time을 반영시켜보기 위해서 해당부분 변경
            image_list.append(image_c)
            '''
        
            if '2skip' in fname:
                    image_c = image_c/8000*2   ## 8000의 기준은 사실 명확하지 않음. 기본적으로 DoS-CaBER로 측정하는 유체의 Frame이 8000frame안에서 종결될 것이라는 가정하에 선정. 이게 너무 높아지면 빨리 끊어지는 유체의 pixel 값이 너무 작아짐.
                    image_list.append(image_c)

            elif i > break_t-int(frame/100*(break_t-start_f)):
              
                            
                if '2skip' in fname:                                      
                    image_c = image_c/8000 *50*2 ## 8000의 기준은 사실 명확하지 않음. 기본적으로 DoS-CaBER로 측정하는 유체의 Frame이 8000frame안에서 종결될 것이라는 가정하에 선정. 이게 너무 높아지면 빨리 끊어지는 유체의 pixel 값이 너무 작아짐.
                    image_list.append(image_c)

                
                else:
                    image_c = image_c/8000*50
                    image_list.append(image_c)
            else:
                image_c = image_c/8000
                image_list.append(image_c)

        image_sum = np.float64(sum(image_list))
        image_sum = np.array(image_sum,dtype=np.float64)
        last_sum = np.float64(sum(image_list[-int(frame/100*(break_t-start_f)):-1]))
        last_sum = np.array(last_sum,dtype=np.float64)
        ys,xs=np.where(image_sum > image_sum.max()-3) ## cropped 범위 및 Center Align 수정
        xset=[]
        for i in set(ys):
           xi=xs[np.where(ys==i)]
           xc=(xi[0]+xi[-1])/2
           xset.append(xc)
        averagecenter=sum(xset)/len(set(ys))
        image_sum = image_sum[140:500,int(averagecenter-200):int(averagecenter+200)]
        last_sum=last_sum[140:500,int(averagecenter-200):int(averagecenter+200)]

        
        
        ## cv2.imwrite(directory+savenumber+'/'+savenumber+'multi_image.png',image_sum)  잠시 변경
        cv2.imwrite('/home/minhyukim/pcalist/'+savenumber+'_multi_image_50_weight_{}frame.png'.format(frame),image_sum)
        cv2.imwrite('/home/minhyukim/pcalist/'+savenumber+'_last_{}frame.png'.format(frame),last_sum)

    def capbridge_terminal(self, image):
        """
        이미지에서 capillary bridge가 끊어져 있는지 여부를 판단합니다.

        Parameters
        ==========

        image : np.ndarray
            실험에서의 이미지입니다.

        Returns
        =======

        bool

        Examples
        ========

        Capillary bridge가 존재할 경우 False를 반환합니다.

        .. plot::
            :include-source:
            :context: reset

            >>> import matplotlib.pyplot as plt
            >>> from mcplexpt.caber.dos import DoSCaBERExperiment
            >>> from mcplexpt.testing import get_samples_path
            >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
            >>> expt = DoSCaBERExperiment(path)
            >>> img1 = expt.get_nth_image(0)
            >>> plt.imshow(img1, cmap='gray') # doctest: +SKIP
            >>> expt.capbridge_broken(img1) # doctest: +SKIP
            False

        .. plot::
            :include-source:
            :context: close-figs

            >>> img2 = expt.get_nth_image(-1)
            >>> plt.imshow(img2, cmap='gray') # doctest: +SKIP
            >>> expt.capbridge_broken(img2) # doctest: +SKIP
            True

        """
        image_check = self.get_nth_image(-1)
        image_check = image_check.download()
        image_c = cv2.Canny(image_check,50,220)
        yss, xss = np.where(image_c == 255)    
        thick_l =[]
        row_l =[]
        for y in set(yss):
            edge_xss = xss[np.where(yss == y)]
            row, thick = (y, edge_xss[-1]-edge_xss[0])
            if y < 9 :
                thick_l.append(thick)
            elif y >= 9 and thick < int(sum(thick_l)/len(thick_l)-10) : ## 노즐 두께에 따라 310이 아닐수 있기때문에 변경필요.
                row_l.append(row)
                break
        nozzle_thick = int(sum(thick_l)/len(thick_l))
        row_roi = row_l[0]
        image=image.download()
        h,w = image.shape
        roi_image = image[0:int(row_roi+1.5*nozzle_thick+10),0:w] ## Neck의 하단부는 roi에서 잘려나감. 조금 더 넓은범위 확인할 수 있으면서 노이즈끼지 않도록 수정필요.

        ret,thresh_image = cv2.threshold(roi_image,130,255,cv2.THRESH_BINARY) ## 약간 빨리 끊어진걸로 판단하는 경향있음. 필요에 따라 임계점 변경 or 3줄이상 모두 White 발생 이런식으로 바꾸는게 좋을 수도 있음.
        
        y_w, x_w = np.where(thresh_image==255)
        y_b, x_b = np.where(thresh_image==0)
        
        y_w_row = set(y_w)
        y_b_row = set(y_b)
        only_w_y_row = set.difference(y_w_row,y_b_row)
        h,w = thresh_image.shape
        

        if len(only_w_y_row) == 0:
            return False
        ## elif len(only_w_y_row) != 0 and max(only_w_y_row) != h and min(only_w_y_row) != 0:
            ## return True 임시로 변경 (BOAS 현상 까지 이미지 병합에 넣으려고)
        elif len(only_w_y_row) > 10 and max(only_w_y_row) != h and min(only_w_y_row) != 0:
            return True        
        
        

class DoSCaBERError(Exception):
    """
    DoS CaBER 분석 시 발생하는 에러를 위한 클래스입니다.
    """
    pass