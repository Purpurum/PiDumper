from io import BytesIO
import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import detectors
from res.style import page_style
import csv
import shutil
import base64

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
    model_ans_seg1 = YOLO("models/cls_ans_seg1.pt") #n
    model_ans_seg2 = YOLO("models/cls_ans_seg2.pt") #n
    model_ans_seg3 = YOLO("models/cls_ans_seg3.pt") #m
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
        dict = {0: 'кирпичи', 1: 'бетон', 2: 'грунт', 3: 'дерево'}
        col1, col2 = st.columns(2)
        with col1:
            on = st.toggle('Boost on')
            if on:
                route = "fast"
            else:
                route = "accurate"
        with col2:
            on = st.toggle('Подключть ансамбль моделей?')
            if on:
                ensemle = True 
            else:
                ensemle = False
        

        uploaded_data = st.file_uploader(label="Выберите файл для распознавания",type=load_type)       

        if uploaded_data is not None:
            file_type = get_file_type(uploaded_data.name)
            if any(file_type == i for i in load_type[:3]):
                image_original = load_img(uploaded_data)
                type = "image"
                #Вывод после картинки,  нолик - просто потому что---------------------------------
                #Вывод номеров классов - тупа словарь ------------------------------------------------
                if route == "fast":
                    col3, col4 = st.columns(2)
                    with col3:
                        st.image(image_original, caption="Оригинальное изображение",width=300)
                    with col4:
                        st.image(detectors.run_model(models, image_original, "detect")[0], caption="Результат детекции",width=300)
                    if ensemle == True:
                        st.image(detectors.run_model(models, image_original, "class_detect")[0], caption="Результат классификации",width=300)
                    else:
                        st.image(detectors.run_model(models, image_original, "cls_det_ansemble3")[0], caption="Результат классификации",width=300)
                else:
                    col3, col4 = st.columns(2)
                    with col3:
                        st.image(image_original, caption="Оригинальное изображение",width=300)
                    with col4:
                        st.image(detectors.run_model(models, image_original, "segment")[0], caption="Результат сегментации",width=300)
                    if ensemle == True:
                        st.image(detectors.run_model(models, image_original, "cls_seg_ansemble3")[0], caption="Результат классификации",width=300)
                    else:
                        st.image(detectors.run_model(models, image_original, "class_segment")[0], caption="Результат классификации",width=670)
            elif file_type == load_type[8]:
                if uploaded_data is not None:
                        type = "zip"
                        results_list, files_list = detectors.run_full_cycle(models=models, data=uploaded_data, type=type, ensemble=ensemle, route=route)
                        #Вывод после видоса, первый нолик - номер в списке, второй нолик - просто потому что--
                        #Вывод номеров классов - тупа словарь ------------------------------------------------
                        
                        data = [['id', 'name', 'result']]
                        
                        for idx, item in enumerate(results_list):
                            data.append([idx, files_list[idx], dict[item]])
                        if st.button('Получить ссылку'):
                            create_and_download_csv(data, "result.csv")
                        

            elif any(file_type == i for i in load_type[3:8]):
                type = "video"
                print(ensemle)
                results_list = detectors.run_full_cycle(models=models, data=uploaded_data, type=type, ensemble=ensemle, route=route)
                #Вывод после видоса, первый нолик - номер в списке, второй нолик - просто потому что--
                #Вывод номеров классов - тупа словарь ------------------------------------------------
                st.text(dict[results_list])
                st.text("--------------------------------------------------------------------")
                #Вывод вероятностей ------------------------------------------------
                st.text(results_list)
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


def create_and_download_csv(data, filename):
    # Create CSV file
    csv_file = open(filename, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerows(data)
    csv_file.close()

    # Read the CSV file in binary mode
    with open(filename, 'rb') as f:
        csv_data = f.read()

    # Create a download button
    b64 = base64.b64encode(csv_data).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    st.markdown(linko, unsafe_allow_html=True)
        
    

def load_img(img_file):
    img = img_file.getvalue()
    image = Image.open(BytesIO(img))
    return image

if __name__ == '__main__':
    models = load_models()
    main()
    
