# =============================================================================
# Author : Tushar Rakshe
# Project OCR based on xml template

#https://github.com/UB-Mannheim/tesseract/wiki # Binary OCR download link
# =============================================================================
import xml.etree.cElementTree as ET
import pandas as pd
import datetime
from pdf2image import convert_from_path
import cv2
import pytesseract
from PIL import Image
import tempfile
import numpy as np
import glob, os
import re
from Image_Alignment import Image_Alignment
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd ='C:\\Users\\214525\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'     
#XML_Filepath = "TestTemplate.xml"
XML_Filepath = "TestTemplate_v2.xml"
#INPUT_IMAGE_PATH = "Guard_ CMS1500 2_page_0_Rotated.jpg"
Reference_Image  = "TestTemplate.jpg"

#XML_Filepath = '0_Sample_Form.xml'
#Reference_Image = "0_Sample_Form.jpg"
PreProcess = True
save= False
save_match = False
method = "SURF"
# =============================================================================
# XML File reading and parsing
# =============================================================================
def Extract_Text(XML_Filepath, INPUT_IMAGE_PATH, Reference_Image, method, PreProcess, save, save_match):  
    Start_Time = datetime.datetime.now()
    dicts = {}    
    Alignment_Start_Time = datetime.datetime.now()
    IMAGE = Image_Alignment(Reference_Image, INPUT_IMAGE_PATH, method = method, save_match = save_match, save= save)    
    Alignment_EndTime = datetime.datetime.now()
    print("Time Taken for Image Registration is : ", Alignment_EndTime - Alignment_Start_Time)
    xml_file = XML_Filepath
    #print(xml_file)
    xml_list = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                 root.find('path').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
                )
        xml_list.append(value)
    column_name = ['filename','path', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    #if Pre_Process == "true":
    #    ALL_TEXT = pytesseract.image_to_string(IMAGE)
    #else:
    #    ALL_TEXT = pytesseract.image_to_string(Pre_Process_IMAGE)   
#    dicts["File_Name"] = INPUT_IMAGE_PATH
    dicts["File_Name"] = INPUT_IMAGE_PATH.split("\\")[-1]    
    for i in xml_df['class'].values:  
        print(i)
        box = xml_df.loc[xml_df['class']==i,['xmin', 'ymin', 'xmax', 'ymax']].values
        box = box.astype('int')
        box =np.reshape(box,(2,2))
        x,y,w,h = cv2.boundingRect(box) 
        temp = IMAGE[y:y+h, x:x+w]   
#        plt.imshow(temp)
        config = ('-l eng --oem 1 --psm 6')
        if PreProcess == True:
            #OUT_IMAGE_PATH = Pre_Process_img_name+'.jpg'
    #        (OUTPUT_IMAGE_PATH,Pre_Process_IMAGE) = Final_Preprocess(INPUT_IMAGE_PATH)
    #        temp = Pre_Process_IMAGE[y:y+h, x:x+w]
#            im_gray = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
            (thresh, result) = cv2.threshold(temp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(result, config=config)
    #        text = pytesseract.image_to_string(Pre_Process_IMAGE(temp)[1]), lang = 'eng')
        else:
    #        temp = IMAGE[y:y+h, x:x+w]
            #text = pytesseract.image_to_string(image_smoothening(convertToGray(temp)), lang = 'eng')
            text = pytesseract.image_to_string(temp,config=config)   
        dicts[i] = text
        
    EndTime = datetime.datetime.now()    
    dicts["Time Taken for Image Registration"] = Alignment_EndTime - Alignment_Start_Time
    dicts["Total Time Taken"] = EndTime - Start_Time
    print("Total Time Taken for Execution:", EndTime - Start_Time)
    return dicts
    