from io import BytesIO
import os
import shutil
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO
import detectors
import utils
from res.style import page_style
import re
import requests
import zipfile

@st.cache_data
def load_models():
    print("loading_models")
    model0 = YOLO("models/seg_n_aug.pt")
    model1 = YOLO("models/cls_seg_garb_aug.pt")
    model2 = YOLO("models/cls_truck.pt")
    model3 = YOLO("models/det_garbage_aug.pt")
    model4 = YOLO("models/cls_correct.pt")
    model5 = YOLO("models/cls_det_garbage_n_aug.pt")
    model_ans_det1 = YOLO("models/cls_ans_det1.pt") #n
    model_ans_det2 = YOLO("models/cls_ans_det2.pt") #n
    model_ans_det3 = YOLO("models/cls_ans_det3.pt") #m
    model_ans_seg1 = YOLO("models/cls_ans_seg2.pt") #n
    model_ans_seg2 = YOLO("models/cls_ans_seg2.pt") #n
    model_ans_seg3 = YOLO("models/cls_ans_seg2.pt") #m
    models_all = [model0, model1, model2, model3, model4, model5, model_ans_det1, model_ans_det2, model_ans_det3, model_ans_seg1, model_ans_seg2, model_ans_seg3]
    return models_all

#Учитывай, что результатов может быть несколько, если было несколько подходящих фреймов в одом видео--------------------------------------
def main():
    #Загрузка стилей
    page_style()
    # Установка заголовка 
    st.title('Классификация строительных отходов')
    #Функция загрузки файла
    def load():
        load_type = ['jpg','png','jpeg','mkv','mp4','mpg','mpeg','mpeg4','zip']
        route = "accurate" #выбор типа цикла 'fast' или 'accurate'.
        ensemle = False #используется ли ансамбль.
        col1, col2 = st.columns(2)
        with col1:
            on = st.toggle('Ускорить? с потерей качества')
            if on:
                route = "fast"
        with col2:
            on = st.toggle('Подключть ансамбль моделей?')
            if on:
                ensemle = True 

        uploaded_data = st.file_uploader(label="Выберите файл для распознавания",type=load_type)       

        if uploaded_data is not None:
            file_type = get_file_type(uploaded_data.name)
            if any(file_type == i for i in load_type[:3]):
                image_original = load_img(uploaded_data)
                type = "image"
                image_cycle, res_cycle = run_cycle(models, image_original, type, ensemle, route)
                #Вывод после картинки,  нолик - просто потому что---------------------------------
                #Вывод номеров классов - тупа словарь ------------------------------------------------
                st.text(res_cycle[0].names)
                st.text("--------------------------------------------------------------------")
                #Вывод вероятностей ------------------------------------------------
                st.text(res_cycle[0].probs)
                st.text("--------------------------------------------------------------------")
                #Вывод чего-то определённого, например топ1
                st.text(res_cycle[0].probs.top1)
                st.text("--------------------------------------------------------------------")
                type(image_cycle)
                st.image(image_cycle)

            elif file_type == load_type[8]:
                if uploaded_data is not None:
                    temp_dir = tempfile.mkdtemp()

                    with zipfile.ZipFile(BytesIO(uploaded_data.getvalue()), 'r') as myzip:
                        myzip.extractall(path=temp_dir)

                    #Всё как у видео, только выводит столько раз подряд, сколько было видео
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        type = "video"
                        results_list = run_cycle(models, file_path, type, ensemle, route)
                        #Вывод после видоса, первый нолик - номер в списке, второй нолик - просто потому что--
                        #Вывод номеров классов - тупа словарь ------------------------------------------------
                        st.text(results_list[0][0].names)
                        st.text("--------------------------------------------------------------------")
                        #Вывод вероятностей ------------------------------------------------
                        st.text(results_list[0][0].probs)
                        st.text("--------------------------------------------------------------------")

                    shutil.rmtree(temp_dir)

            elif any(file_type == i for i in load_type[3:8]):
                type = "video"
                results_list = run_cycle(models, uploaded_data, type, ensemle, route)
                #Вывод после видоса, первый нолик - номер в списке, второй нолик - просто потому что--
                #Вывод номеров классов - тупа словарь ------------------------------------------------
                st.text(results_list[0][0].names)
                st.text("--------------------------------------------------------------------")
                #Вывод вероятностей ------------------------------------------------
                st.text(results_list[0][0].probs)
                st.text("--------------------------------------------------------------------")
            else:
                st.text("Ошибка чтения файла, возможно неподходящий формат\n(убедитесь что в названии файла нет точек и специальных символов)")          
        else:
            return None
    #Функция для получения расширения файла
    def get_file_type(file):
        return os.path.splitext(file)[1][1:]   
    
    load()
    #load_camera()
def run_cycle(models, image_original, type, ensemle , route):
    if type == "image":
        image_detected, results = detectors.run_full_cycle(models, image_original, type, ensemle , route)
        return image_detected, results
    elif type == "video":
        results_list = detectors.run_full_cycle(models, image_original, type, ensemle , route)
        return results_list
    
def detect_objects(models, image, type = "segment"):
    image_detected, results = detectors.run_model(models, image, type)
    return image_detected, results

def load_img(img_file):
    img = img_file.getvalue()
    image = Image.open(BytesIO(img))
    return image

if __name__ == '__main__':
    models = load_models()
    main()
    

if __name__ == '__main__':
    models = load_models()
    main()
    
