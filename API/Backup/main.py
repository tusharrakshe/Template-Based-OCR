# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:10:43 2019

# =============================================================================
# @author: Tushar Rakshe
# =============================================================================
"""
from flask import Flask, render_template, jsonify,request,redirect,url_for, flash  
from flask_cors import CORS 
import pandas as pd
import mysql.connector
from mysql.connector import Error
import json
import csv, re, collections
from collections import Counter
import nltk
import sys
import io
import os
import urllib.request
from werkzeug.utils import secure_filename
from PyPDF2 import PdfFileReader
from pprint import pprint
import time
import numpy as np
import glob, os
import argparse
import xml.etree.ElementTree as ET
import cv2
import pytesseract
from PIL import Image
import tempfile
from pdf2image import convert_from_path
# importing all the required modules
import PyPDF2
import time
import datetime
import pandas as pd
import re
import textile
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

from Extract_Text import *
from Accuracy_Evaluation import MisSpell_Count

UPLOAD_FOLDER = 'E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\API\\static\\Uploads'
outputimgdir = 'E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\API\\static\\Uploads'

app = Flask(__name__, template_folder='E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\API\\templates')
app.secret_key = '12345'
CORS(app) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
connection = ""

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180
size = None

def get_size_of_scaled_image(im):
    global size
    if size is None:
        length_x, width_y = im.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
    return size           


start = time.process_time()
def set_image_dpi(file_path):
    im = Image.open(file_path)
    # size = (1800, 1800)
    size = get_size_of_scaled_image(im)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))  # best for OCR
    return temp_filename

def Remove_lines(IMAGE):
    image = cv2.imread(IMAGE)
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    ## Remove horizontal
    # Defining a kernel length
    kernel_length = np.array(image).shape[1]//40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)
    
    # Repair after Horizontal line removal image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
    
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel_length))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)
    
    # Repair after Horizontal line removal image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
    
    
    #convert image to black amd white
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    (thresh, result) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return result

def convertToGray(img):
    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def process_image_for_ocr(file_path):
    temp_filename = set_image_dpi(file_path)
    im_new = Remove_lines(temp_filename)
    return im_new

#def getMetadata(path):
#    image = Image.open(path)
#    try:
#        dpi = image.info["dpi"]
#    except:
#        dpi = 'Not_defined'
#    #print(image.size)
#    #print(image.info)
#    width = image.size[0]
#    #print(width)
#    height = image.size[1]
#    #print(height)
#    Resolution = (width*height)/1000000
#    return height,width,Resolution,dpi

def getMetadata(path):
    parser = createParser(path)
    metadata = extractMetadata(parser)
    METADATA  = metadata.exportDictionary()
    METADATA = METADATA['Metadata']
    width = [int(s) for s in METADATA['Image width'].split() if s.isdigit()][0]
    height = [int(s) for s in METADATA['Image height'].split() if s.isdigit()][0]
    Resolution = (width*height)/1000000
    Bits_Per_pixel = METADATA['Bits/pixel']
    return height,width,Resolution,Bits_Per_pixel
#
def rotate_image(path, center = None, scale = 1.0):
    # Reading_of_original_image
    image = cv2.imread(path)
    # angle throgh which original image to be rotated
    angle= 360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Rotated Image
    rotated = cv2.warpAffine(image, M, (w, h))       
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename,rotated)
    return temp_filename

def rotate_text(image, center = None, scale = 1.0):
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]   
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1] 
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    elif angle == 0:
        angle =360          
    else:
        angle = -angle                       
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename,rotated)
    return temp_filename    

# Function to read, preprocess, extract and save the results
def OCR(file_name,OUTPUT_Dir,Pre_Process):
    start_time = time.time()
    INPUT_IMAGE_PATH = os.path.join(outputimgdir, file_name)
    FILE_TYPE = INPUT_IMAGE_PATH.split(sep = '.')[-1]

    if FILE_TYPE == 'pdf':
        pages = convert_from_path(INPUT_IMAGE_PATH)
        INPUT_IMAGE_PATH = os.path.join(outputimgdir, file_name.split(sep='.')[0])+'.jpg'
        for i, page in enumerate(pages):
#            print(i)
            if i==0:
                page.save(INPUT_IMAGE_PATH)

#    Pre_Process_IMAGE = process_image_for_ocr(INPUT_IMAGE_PATH)
#    dt = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
#    Pre_Process_img_name = '_'.join(["Pre_Process_IMAGE",dt])
#    cv2.imwrite(OUTPUT_Dir+Pre_Process_img_name+'.jpg',Pre_Process_IMAGE)
    #IMAGE = cv2.imread(set_image_dpi(IMAGE_PATH))
    IMAGE = cv2.imread(INPUT_IMAGE_PATH)


# =============================================================================
#     Text Extraction OCR
# =============================================================================
    
    if Pre_Process == "true":
        try:
            temp_file_path = rotate_image(INPUT_IMAGE_PATH)
        except:
            pass
        if temp_file_path:
            Pre_Process_IMAGE = process_image_for_ocr(temp_file_path)
        else:
            Pre_Process_IMAGE = process_image_for_ocr(INPUT_IMAGE_PATH)   
            
        dt = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
        Pre_Process_img_name = '_'.join(["Pre_Process_IMAGE",dt])
        cv2.imwrite(os.path.join(OUTPUT_Dir,Pre_Process_img_name)+'.jpg',Pre_Process_IMAGE)
        ALL_TEXT = pytesseract.image_to_string(Pre_Process_IMAGE, lang = 'eng')
        
        OUT_IMAGE_PATH = Pre_Process_img_name+'.jpg'
    
    else:
        center = None
        scale = 1.0
        try:
            angle= 360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(IMAGE)).group(0))
            (h, w) = IMAGE.shape[:2]
            
            if center is None:
                center = (w / 2, h / 2)    
            # Perform the rotation
            M = cv2.getRotationMatrix2D(center, angle, scale)# Rotated Image
            IMAGE = cv2.warpAffine(IMAGE, M, (w, h))   
        except:
            pass
        file_name=file_name.split(sep='.')[0]+'_1.'+file_name.split(sep='.')[1]
        cv2.imwrite(os.path.join(OUTPUT_Dir,file_name),IMAGE)
        
        ALL_TEXT = pytesseract.image_to_string(IMAGE,lang = 'eng') 
        OUT_IMAGE_PATH = file_name
        

    TEXT_FILE = os.path.join(OUTPUT_Dir,file_name.split(sep = '.')[:-1][0]+'.txt')
    HTML_TEXT_FILE = os.path.join(OUTPUT_Dir,file_name.split(sep = '.')[:-1][0]+'_HTML'+'.txt')    
    with open(TEXT_FILE,mode ='w') as file:
        file.write(ALL_TEXT)        
    Percentage_Extraction,Error_Per = MisSpell_Count(TEXT_FILE) 
    with open(HTML_TEXT_FILE,mode ='w') as file:
        file.write("<pre>" + ALL_TEXT + "</pre>")    
# =============================================================================
#     Table Reading EBOS
# =============================================================================
#    Keywords = ['Name','Patient','Patient Name','Elderplan ID','Claim No','Provider Name','Provider NPI','Acct','Proc','ID','Number','Amount','Date']       
#
#    with open(TEXT_FILE, 'r') as in_file:
#        stripped = (re.sub('[^A-Za-z0-9\$\/\-\s\.\:]+', '', line).strip() for line in in_file  if len([i for i in line if i =='$']) > 5 or True in [Keyword in line for Keyword in Keywords])
#    #    lines = (line.split(' ') if (len([i for i in line if i =='$']) > 5 or 'Proc' in line) else line.split(",") for line in stripped if line)           
#        lines = (line.split(' ') if (len([i for i in line if i =='$']) > 5 ) else line.split(",") for line in stripped if line)           #    lines = (line.split(" ") for line in stripped if len([i for i in line if i =='$']) > 5 )
#        with open(TEXT_FILE.split('.')[0]+'.csv', 'w') as out_file:
#            writer = csv.writer(out_file)
#            writer.writerows(lines)
#    
#    Struct_DATA = pd.read_csv(TEXT_FILE.split('.')[0]+'.csv', names = range(0,20), engine='python')
#    Struct_DATA = Struct_DATA.dropna(axis =1,how='all')
#    Struct_DATA = Struct_DATA.fillna("")
    Struct_DATA = pd.DataFrame()
    with open(HTML_TEXT_FILE, "r") as file:
        ALL_TEXT = file.read()
        
#    OUT =  Unstructure2Struct(TEXT_FILE) 
    Dicts = Extract_Text(XML_Filepath, INPUT_IMAGE_PATH, Reference_Image, method, Pre_Process, save, save_match)
    OUT = pd.DataFrame({"Keyfields" : list(Dicts.keys()), "Extracted_Values": list(Dicts.values())})
    OUT = OUT[0:26]
    OUT.to_csv(OUTPUT_Dir+'Output_CSV.csv', index=False)

    #OUT_CSV_PATH = OUTPUT_Dir+'Output_CSV.csv'
    INPUT_IMAGE = os.path.join(outputimgdir, OUT_IMAGE_PATH)
    h,w ,r, dpi= getMetadata(INPUT_IMAGE)
    Computation_Time = round((time.time() - start_time),2)
# =============================================================================
#    Banner
# =============================================================================
#    Banner_Out = ribbon(INPUT_IMAGE_PATH)   ### Making it Comment for fast execution
    Banner_Out = None
#    print("--- %s seconds ---" % end_time)
#    Statistics = pd.DataFrame({"Extraction %":[Percentage_Extraction*100],
#                               "Error %":[Error_Per],
#                               "Computation_Time":[Computation_Time]})
#    print(Statistics)
    return OUT_IMAGE_PATH,OUT,h,w ,r,ALL_TEXT,dpi,Struct_DATA,Percentage_Extraction,Error_Per,Computation_Time,Banner_Out
   
# Creating an empty Dataframe with column names only
#Final_Emoji_Result = pd.DataFrame(columns=['Text', 'Emoji', 'Meaning', 'Location', 'Score'])
colnames =['Text', 'Emoji', 'Meaning', 'Location', 'Score']
appended_data = []
#Mathsdf = pd.read_excel('D:\emoji-sentiment-data\MathematicalSymbols.xlsx', index_col=None) 
#mathtestsymbol = Mathsdf[['Symbol','Name']]
#mathtestsymboldf = mathtestsymbol.groupby(['Symbol'])['Name'].apply(', '.join).reset_index()
# Creating an empty dictionary 
#myDict = {} 
## Filling the empty dictionary 
#for i in range(len(mathtestsymboldf)) :
#    key = mathtestsymboldf['Symbol'][i]
#    value = mathtestsymboldf['Name'][i]
#    myDict[key] = [value]

pytesseract.pytesseract.tesseract_cmd ='C:\\Users\\214525\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'     

@app.route("/", methods=['GET', 'POST'])
@app.route("/chartjs", methods=['GET', 'POST'])
def chartjs():
    try:
        pdf_path = '/static/Uploads/'
        pdf_filename = 'passport_form_of_raykar.pdf'
        pdf_location = os.path.join(pdf_path,pdf_filename)
#        EntireData = Final_Emoji_Result.reindex(index=Final_Emoji_Result.index[::-1])
#        EntireData = json.loads(EntireData.to_json(orient='records'))

        if request.method == 'POST':
                    file = request.files['file'] if request.files.get('file') else None
                    if file is None:
                        print("reached at condition1")
                        Pre_Process = request.form.get('ip1')
                        file_name = request.form.get('name')
                        file_name = file_name.split(sep='.')[0]+'.jpg'
#                        file_name = ''.join([file_name,st])
                        print("preprocess: ",Pre_Process)
                        print("Img name: ", file_name)
                        img_path = '/static/Uploads/'
                        outputimgdir = 'E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\API\\static\\Uploads'
#                        INPUT_IMAGE_PATH = os.path.join(outputimgdir, file_name)
                        OUT_IMAGE,OUT_CSV,height,width,resolution,ALL_TEXT,dpi,Struct_DATA,Percentage_Extraction,Error_Per,Computation_Time,Banner_Out = OCR(file_name,outputimgdir,Pre_Process)
                        OUT_IMAGE_PATH = os.path.join(img_path, OUT_IMAGE)
                        print("OUT_IMAGE_PATH: ",OUT_IMAGE_PATH)
                        colnames = OUT_CSV.columns.values
                        colnames=colnames.astype(str)
                        colnames = json.loads(json.dumps(colnames.tolist()))
                        EntireData = json.loads(OUT_CSV.to_json(orient='records'))
                        colnames1 = Struct_DATA.columns.values
                        colnames1=colnames1.astype(str)
                        colnames1 = json.loads(json.dumps(colnames1.tolist()))
                        EntireData1 = json.loads(Struct_DATA.to_json(orient='records'))
#                        print("OUT_CSV_PATH: ",OUT_CSV)
                        result = "text of the uploaded file"
                        print("condition1: "+result)
#                        ALL_TEXT = textile.textile(ALL_TEXT)
                        return jsonify(result=ALL_TEXT,EntireData=EntireData,imgfile=OUT_IMAGE_PATH,colnames=colnames,height=height,width=width,resolution=resolution,dpi=dpi,EntireData1=EntireData1,colnames1=colnames1,Percentage_Extraction=Percentage_Extraction,Error_Per=Error_Per,Computation_Time=Computation_Time,result1=Banner_Out)
                    
                    if file:
                        print("reached at condition2")
                        file_name = secure_filename(file.filename)
                        print("file_name: "+file_name)
#                        file_name = "audit1.pdf"
                        
                        img_path = '/static/Uploads/'
                        outputimgdir = 'E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\API\\static\\Uploads'
                        file.save(os.path.join(outputimgdir, file_name))
                        imgfile = os.path.join(img_path, file_name)
                        INPUT_IMAGE_PATH = os.path.join(outputimgdir, file_name)
                        print("INPUT_IMAGE_PATH: ",INPUT_IMAGE_PATH)
                        FILE_TYPE = INPUT_IMAGE_PATH.split(sep = '.')[-1]
                        print("FILE_TYPE: ",FILE_TYPE)
                        if FILE_TYPE == 'pdf':
                            pages = convert_from_path(INPUT_IMAGE_PATH,500)
                            INPUT_IMAGE_PATH = os.path.join(outputimgdir, file_name.split(sep='.')[0])+'.jpg'
                            imgfile = os.path.join(img_path, file_name.split(sep='.')[0])+'.jpg'
#                            for page in pages:
#                                page.save(INPUT_IMAGE_PATH)
                            for i, page in enumerate(pages):
                                #print(i)
                                if i==0:
                                    page.save(INPUT_IMAGE_PATH)                           
#                        read_path = "E:/Character Recognition/Demo/" # path folder where the input pdf files are saved.
#                        write_path = "E:/Tushar/Projects/Image_Analytics/OCR/Frontend/OCRfrontend/AdminLTE-master/static/Uploads/OCR_result.xlsx" # Excel file where output need to be saved.
#                        OUT_IMAGE_PATH,OUT_CSV_PATH,height,width,resolution = OCR(INPUT_IMAGE_PATH,OUTPUT_Dir,Pre_Process)
                        print(time.process_time() - start)
                        print("output imgfile",imgfile)
                        file_name1=file_name.split(sep='.')[0]+'.jpg'
                        INPUT_IMAGE = os.path.join(outputimgdir, file_name1)
                        height,width,Resolution,dpi= getMetadata(INPUT_IMAGE)
#                        colnames = EntireData.columns.values
#                        colnames = json.loads(json.dumps(colnames.tolist()))
#                        print("Colnames: ",colnames)
#                        EntireData = json.loads(EntireData.to_json(orient='records'))
                        name = file_name
                        dept = "DART"
                        print("condition2: "+name)
#                        print(imgfile)
                        print(dpi)
                        print(Resolution)
                        return jsonify(filename=name,imgfile=imgfile,dept=dept,height=height,width=width,resolution=Resolution,dpi=dpi)
#                        return jsonify(name=name,imgfile=imgfile,dept=dept)
                  
                    else:
                        print("reached at condition3")
                        result = "Allowed file types are wav, mp3"
                        print("condition3: "+result)
                        return jsonify(colnames=colnames)
                
      
        return render_template("chartjs.html",result='No data available in text', pdf_location=pdf_location)
    except Exception as e:
        return(e)
 
   

if __name__ == "__main__":
    app.run(debug=True)