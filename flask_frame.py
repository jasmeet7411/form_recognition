#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:27:16 2018

@author: meet
"""

import os
from flask import Flask, render_template, request, send_file
#import funcs_mform as mform
import processing_input as proc_inp


#cll,cll_digit,X_train,X_train_digit=mform.load_models()

app = Flask(__name__)
 
@app.route('/')
def form():
    return render_template('form.html')
    
    
root_path = os.path.dirname(os.path.abspath(__file__))

target = os.path.join(root_path,'static')

@app.route('/val', methods=['GET','POST'])
def val():
    values=request.files.getlist('img')
    strr=''
    
    for i in values:
        fname=i.filename
        fpath='/'.join([target,fname])
        i.save(fpath)
        strr=strr+str(fname)+','
        
    inm=strr.split(sep=',')
    inm=inm[:-1]
    
    proc_inp.process_input(inm)
    """
    page_list,res_file=mform.crop_pages(inm)

    for page_seg in page_list:
        seg_list=mform.crop_segs(page_seg)
        
        for seg_part in seg_list:
            mform.seg_to_binary_values_and_pred(seg_part,cll,cll_digit,X_train,X_train_digit,res_file)
    """
    f=open('result_file.txt','r')
    txt_line=[]
    for i in values:
        txt_line.append(f.readline())
    f.close()
    len_val=len(values)
   
    return render_template('result_show.html', inm=inm, txt_line=txt_line,len_val=len_val)
    
    
@app.route('/return-file')
def return_file():
    return send_file('/home/meet/Documents/m_form/result_file.txt', attachment_filename='result_file.txt',cache_timeout=5)    

if (__name__ == "__main__"):
    app.run(port = 5000)
    
        
