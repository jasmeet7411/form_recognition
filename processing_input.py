#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:53:05 2018

@author: meet
"""

#Importing libraries
def process_input(input_path):
    import numpy as np
    import cv2
    from sklearn.externals import joblib
    
    
    #Loading the saved model
    cll=joblib.load('minor_proj_char_model.pkl')
    
    cll_digit=joblib.load('minor_proj_char_model_digits.pkl')
    
    #loading fit values
    X_train=joblib.load('minor_proj_xtrain_fitval.pkl')
    
    X_train_digit=joblib.load('minor_proj_xtrain_fitval_digits.pkl')
    
    
    #creating a file in which results will be saved
    res_file=open('result_file.txt','w')
    
    #Loading input paths
    """
    i_path=open('input_paths.txt','r')
    ii=i_path.readline()
    i_path.close()
    
    input_path=ii.split(sep=',')
    """
    page_list=[]
    for ffff in range(len(input_path)):
        img=cv2.imread('static/'+input_path[ffff])
        #grayscale
        rez=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #interpolation(resizing)
        re_img=cv2.resize(rez,(900,1200),fx=0.3,fy=0.3,interpolation=cv2.INTER_LANCZOS4)
        rez2=re_img.copy()
        
        #blurring
        th_blur=cv2.bilateralFilter(re_img,5,10,30)
        
        #canny edge detection {MOST COMMON AND USEFUL}
        tr=cv2.Canny(th_blur,100,200)
    
        #contours
        __,contours,hierarchy=cv2.findContours(tr.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        srt=[]
        for cc in contours:
            if cv2.contourArea(cc) >=410:
                srt.append(cc)
        
        
        #sorting by area
        sort_cnt=sorted(srt,key=cv2.contourArea,reverse=True)
        
        def transform(pos):
        # This function is used to find the corners of the object and the dimensions of the object
            pts=[]
            n=len(pos)
            for i in range(n):
                pts.append(list(pos[i][0]))
               
            sums={}
            diffs={}
            tl=tr=bl=br=0
            for i in pts:
                x=i[0]
                y=i[1]
                sum=x+y
                diff=y-x
                sums[sum]=i
                diffs[diff]=i
            sums=sorted(sums.items())
            diffs=sorted(diffs.items())
            n=len(sums)
            rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
            #      top-left   top-right   bottom-left   bottom-right
           
            h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)     #height of left side
            h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)     #height of right side
            h=max(h1,h2)
           
            w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)     #width of upper side
            w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)     #width of lower side
            w=max(w1,w2)
           
            return int(w),int(h),rect
    
        max_area=0
        pos=0
        for i in sort_cnt:
            area=cv2.contourArea(i)
            if area>max_area:
                max_area=area
                pos=i
        peri=cv2.arcLength(pos,True)
        approx=cv2.approxPolyDP(pos,0.02*peri,True)
         
        size=img.shape
        w,h,arr=transform(approx)
         
        pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts1=np.float32(arr)
        M=cv2.getPerspectiveTransform(pts1,pts2)
        dst=cv2.warpPerspective(rez2,M,(w,h))
        
        
        img_name='seg_page'+str(ffff)+'.png'
        cv2.imwrite(img_name,dst)
        page_list.append(img_name)
        
    
    for ffff in page_list:
        segment_list=[]
        img=cv2.imread(ffff)
        rez=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        re_img=cv2.resize(rez,(900,1200),fx=1,fy=1,interpolation=cv2.INTER_LANCZOS4)
        
        rez2=re_img.copy()
        
        
        #sharpening
        th_blur=cv2.GaussianBlur(rez2,(7,7),3) #anialiasing type,preserve edges
        shrp=cv2.addWeighted(rez2, 1.5, th_blur, -0.2,0, th_blur)
        
        
        crp_title=shrp[195:228,170:310]
        img_name='det_title.png'
        cv2.imwrite(img_name,crp_title)
        segment_list.append(img_name)
        #if signature also required then 117:520 if not then 117:430
        crp_detail=shrp[269:380,177:640]
        
        #0-124,124-215,215-312 for 600*800 and hh=26
        
        #0-185,185-323,323-464 for 900*1200 and hh=37
        
        for j in range(3):
            hh=37
            hh=j*hh
            for i in range(3):
                crp_d=[]
                if i==0:
                    w1=5
                    w2=185
                if i==1:
                    w1=190
                    w2=323
                if i==2:
                    w1=323
                    w2=464
                crp_d=crp_detail[5+hh:37+hh,w1:w2]
                img_name='det'+str(j)+str(i)+'.png'
                cv2.imwrite(img_name,crp_d)
                segment_list.append(img_name)
                
        
        crp_pname=shrp[400:432,190:764]
        img_name='det_pname.png'
        cv2.imwrite(img_name,crp_pname)
        segment_list.append(img_name)
        
        for ff_seg in range(len(segment_list)):
            img=cv2.imread(segment_list[ff_seg])
            rez=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            rez2=rez.copy()
            
            
            #sharpening
            th_blur=cv2.bilateralFilter(rez,5,10,30)
            shrp=cv2.addWeighted(rez, 1.0, th_blur, -0.01,0, th_blur)
            
            
            #thresholding
            th=cv2.adaptiveThreshold(shrp,255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV,blockSize=35,C=9)
                                #blocksize need to be odd
                                
            
            #contours
            __,contours,hierarchy=cv2.findContours(th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            
            srt=[]
            for cc in contours:
                ca=cv2.contourArea(cc)
                #print(ca)
                if ca <=150.0 and ca >= 10.0:
                    srt.append(cc)
            
            
            #sorting by spatial position
            def x_coord(cont):
                    M=cv2.moments(cont)
                    return (int(M['m10']/M['m00']))
            
            pos_sortcnt=sorted(srt,key=x_coord,reverse=False)
            
            rez2=th.copy()
            
            word_val=np.zeros((28,28),dtype=np.uint8).ravel().reshape(1,-1)
            for (sr,xx) in enumerate(pos_sortcnt):
                char_val=[]
                x,y,w,h=cv2.boundingRect(xx)
                cv2.rectangle(rez2,(x,y),(x+w,y+h),255,1)
                crp_img=th[y:y+h,x:x+w]
                crp_img=cv2.resize(crp_img,(20,20),interpolation=cv2.INTER_AREA)
                tp=bt=lf=rt=4
                """
                if h<=28:
                    tp=int((28-h)/2)
                    bt=28-(tp+h)
                if w<=28:
                    lf=int((28-w)/2)
                    rt=28-(lf+w)
                """
                chk=cv2.copyMakeBorder(crp_img,tp,bt,lf,rt,0)
                #chk=cv2.resize(chk,(28,28),interpolation=cv2.INTER_AREA)
                th10=cv2.adaptiveThreshold(chk,255,adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=45,C=0)
                char_val=th10.ravel().reshape(1,-1)
                word_val=np.concatenate((word_val,char_val),axis=0)
                
            word_val=word_val[1:,:]
            #prediction
            x_val_float=np.array(word_val,dtype=np.float32)
            model_check=[1,2,4,5,7,8]
            if ff_seg in model_check:
                x_val_float=X_train_digit.transform(x_val_float)
                y_pred = cll_digit.predict(x_val_float)
            else:
                x_val_float=X_train.transform(x_val_float)
                y_pred = cll.predict(x_val_float)
                
            res_strr=''
            for y_pred_values in y_pred:
                res_chr=int(y_pred_values)
                if  res_chr>=10:
                    res_strr=res_strr+str(chr(res_chr))
                else:
                    res_strr=res_strr+str(res_chr)
            if ff_seg<len(segment_list)-1:
                res_file.write(res_strr+',')
            else:
                res_file.write(res_strr+'\n')
            
    
    
    res_file.close()