"""
DoS CaBER 실험 이미지를 분석하기 위한 모듈입니다.

"""
import cv2
import numpy as np
import scipy.optimize as optimize
from mcplexpt.core import ExperimentTIFF


class DoSCaBERExperiment_modified(ExperimentTIFF):
    """
    DoS CaBER 실험 결과 TIFF 이미지 파일을 추상화하는 클래스입니다.

    Examples
    ========

    >>> from mcplexpt.caber.dos import DoSCaBERExperiment
    >>> from mcplexpt.testing import get_samples_path
    >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
    >>> expt = DoSCaBERExperiment(path)

    """
    def capbridge_start(self, image): ## 설명 '''~~''' 넣어야 함.

        image_check = self.get_nth_image(-1)
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

    def Dos_CaBER_fixed_min(self,tifname,savenumber):
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
        import pandas as pd
        import matplotlib.pyplot as plt
        from mcplexpt.caber.dos import DoSCaBERExperiment
        from mcplexpt.testing import get_samples_path
        path = get_samples_path("caber", "dos", tifname)
        expt = DoSCaBERExperiment(path)

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
            fps = 10000
            t = i /fps
            time.append(t)
        data = {'time':time, 'Start':ret_s, 'Break':ret, 'width(fixed)':ret1, 'width(min)':ret2, 'widht(min_mov)' :ret3}
        data = pd.DataFrame(data)
        data.to_excel(excel_writer= 'DoS data_{}.xlsx'.format(savenumber))


        ## 해당부분을 다른 함수(모듈)로 구성하는게 좋을듯. 엑셀에서 데이터 불러오도록 하기.
        
    def Dos_CaBER_VE_analysis(self,tifname):
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

        Measurement_data = pd.read_excel('{}\{}'.format(os.getcwd(),tifname),header=0,index_col=0)
        time = Measurement_data['time'].values.tolist()
        ret1 = Measurement_data['widht(fixex)'].values.tolist()
        ret2 = Measurement_data['widht(min)'].values.tolist()
        ret3 = Measurement_data['width(min_mov'].values.tolist()

        ret_s =  np.array(ret3)
        index_s = np.where(ret_s == 'None')
        index_ss = list(index_s[0])
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

        '''
        EC 영역 구분하기
        '''
        #이동평균(Moving Average)
        Y1data= []
        for y in range(len(y3data_log)):
            if y > 29:
                Y = 0
                for f in range(0,30):
                    Y = Y + y3data_log[y-f]
                Y1data.append(Y/30)
            elif y ==0:
                Y1data.append(y3data_log[0])
            else:
                Y = 0
                for f in range(0,y):
                    Y = Y + y3data_log[y-f]
                Y1data.append(Y/len(range(0,y)))
        Y1data_n = np.array(Y1data)
        plt.subplot(2,2,1)
        plt.plot(xdata,Y1data_n,"bo")
        plt.title('min_sort_data_raw')
        plt.ylabel('R/R0')
        plt.xlabel('time')


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
        plt.subplot(2,2,2)
        plt.plot(xdata,Y2data_n,"bo")
        plt.title('min_sort_data_diff')
        plt.ylabel('dR/dR0')
        plt.xlabel('time')

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
        plt.subplot(2,2,3)
        plt.plot(xdata,Y3data_n,"bo")
        plt.title('min_sort_data_diff_diff')
        plt.ylabel('d2R/d2R0')
        plt.xlabel('time')

        ## log(y)의 선형구간(기울기 변화 없는 구간) 선정
        Y3data_n_abs = []
        try:
            for i in Y3data:
                result = int(abs(i))
                Y3data_n_abs.append(result)
        except OverflowError:
            check = len(Y3data) - len(Y3data_n_abs)
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
                        
        lenth_n = np.array(lenth)
        index_lenth = np.where(lenth_n == max(lenth_n[10:])) ## 그려지는 그래프에 따라 VC구간이 Linear하게 계산될 수 있음.
        index_lenth_real = index_lenth[0][0] 

        IC_time = sum(lenth[:index_lenth_real])
        FE_time = sum(lenth[:index_lenth_real+1])
        EC_region_xdata = xdata[IC_time:FE_time] 
        EC_region_y1data = y1data[IC_time:FE_time] 
        EC_region_y1data_log = y1data_log[IC_time:FE_time] 
        EC_region_y2data = y2data[IC_time:FE_time] 
        EC_region_y2data_log = y2data_log[IC_time:FE_time]
        EC_region_y3data = y3data[IC_time:FE_time] 
        EC_region_y3data_log = y3data_log[IC_time:FE_time]
    
        expt = EC_region_xdata
        expy1 = EC_region_y1data
        expy2 = EC_region_y2data
        expy3 = EC_region_y3data

        model = lambda expt, a,b,c : -(expt-c)/3/b+np.log(a**(1/3))
        popt1, pcov1= optimize.curve_fit(model, expt, np.log(expy1)) 
        popt2, pcov2= optimize.curve_fit(model, expt, np.log(expy2)) 
        popt3, pcov3= optimize.curve_fit(model, expt, np.log(expy3)) 
        fig, ax1 = plt.subplots(figsize=(5,5), dpi=80)
        ax1.semilogy(expt, expy1, "bo")
        ax1.plot(expt, np.exp(model(expt,*popt1)),"r-", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt1))
        plt.legend(fontsize=13) 
        plt.title('fiexd_measure')
        fig, ax2 = plt.subplots(figsize=(5,5), dpi=80)
        ax2.semilogy(expt, expy2, "bo")
        ax2.plot(expt, np.exp(model(expt,*popt2)),"r-", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt2))
        plt.legend(fontsize=13) 
        plt.title('min_measure')
        fig, ax3 = plt.subplots(figsize=(5,5), dpi=80)
        ax3.semilogy(expt, expy3, "bo")
        ax3.plot(expt, np.exp(model(expt,*popt3)),"r-", label="fit: a={:5.4f},b={:5.4f},c={:5.4f}".format(*popt3))
        plt.legend(fontsize=13) 
        plt.title('min_sort_measure')

        tf = xdata[FE_time-1] - xdata[IC_time] 
        
    #Extensional viscosity
    
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

        surface_tension = 63.2   #측정물질에따라 변경필요.(jason이랑 연결하기)
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


class DoSCaBERError(Exception):
    """
    DoS CaBER 분석 시 발생하는 에러를 위한 클래스입니다.
    """
    pass
