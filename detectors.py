from PIL import Image
from collections import Counter
import utils
import cv2
import numpy as np
import os
import shutil

def get_mask(results):
    if(results[0].masks is not None):
            mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
            mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
            h2, w2, c2 = results[0].orig_img.shape
            mask = cv2.resize(mask_3channel, (w2, h2))
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0,0,0])
            upper_black = np.array([0,0,1])
            mask = cv2.inRange(mask, lower_black, upper_black)
            mask = cv2.bitwise_not(mask)
            masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)
            return masked
    
def get_crop(results):
    if(results[0].boxes is not None):
        for box in results[0].boxes.xyxy.tolist():
            x1, y1, x2, y2 = box
            ultralytics_crop_object = results[0].orig_img[int(y1):int(y2), int(x1):int(x2)]
            black_background = np.zeros_like(results[0].orig_img)
            black_background[int(y1):int(y2), int(x1):int(x2)] = ultralytics_crop_object
            return black_background

def run_model(models, image, type = "segment"):
    """
    Функция для детекции изображения одиночной моделью.
    """
    if type == "segment":
        #print("Сегментирую")
        results = models[0](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_segment":
        #print("Классифицирую мусор насегментированный")
        results = models[1](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_truck":
        #print("Классифицирую наличие грузовика")
        results = models[2](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "detect":
        #print("Детекчу")
        results = models[3](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_correct":
       # print("Проверяю корректность изображения")
        results = models[4](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_detect":
        #print("Классифицирую надетекченное")
        results = models[5](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "cls_det_ansemble1":
        #print("Классифицирую надетекченное в рамках ансамбля")
        results = models[6](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "cls_det_ansemble2":
        #print("Классифицирую надетекченное в рамках ансамбля")
        results = models[7](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "cls_det_ansemble3":
        #print("Классифицирую надетекченное в рамках ансамбля")
        results = models[8](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "cls_seg_ansemble1":
        #print("Классифицирую сегментированное в рамках ансамбля")
        results = models[9](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "cls_seg_ansemble2":
        #print("Классифицирую сегментированное в рамках ансамбля")
        results = models[10](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "cls_seg_ansemble3":
        #print("Классифицирую сегментированное в рамках ансамбля")
        results = models[11](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    else:
        print("Некорректный ввод типа модели!")

def run_full_cycle(models, data, type, ensemle = False, route = "accurate"):
    """
    Функция представляет собой полный цикл обработки данных, будь то видео или изображение

    Parameters:
    models: пачка моделей.
    data: данные.
    ensemble: используется ли ансамбль.
    route: выбор типа цикла 'fast' или 'accurate'.
    """
    def cycle_accurate(image, ansemble = True):
        result_img, results = run_model(models, image, "class_truck")
        if results[0].probs.top1 == 1 or results[0].probs.top1 == "1":
            result_img_seg, results_seg = run_model(models, image, "segment")
            result_img_mask = get_mask(results_seg)
            result_img_cls, results_cls = run_model(models, result_img_mask, "class_segment")
            if ansemble != True:
                return result_img_cls, results_cls
        else:
            pass
            
    def cycle_fast(image, ansemble = False):
        result_img, results = run_model(models, image, "class_truck")
        if results[0].probs.top1 == 1 or results[0].probs.top1 == "1":
            result_img_det, results_det = run_model(models, image, "detect")
            result_img_mask = get_crop(results_det)
            result_img_cls, results_cls = run_model(models, result_img_mask, "class_detect")
            if ansemble != True:
                return result_img_cls, results_cls
        else:
             pass

    if type == "image":
        if route == "accurate":
            return cycle_accurate(data)
        else:
            return cycle_fast(data)
        
    elif type == "video":
        frames_list = []
        results_list = []
        #Временной интервал соответствующий заданию
        frames_folder, fps = utils.video_to_frames(data, 110, 130, 20)
        if not os.listdir(frames_folder):
            print(f"The directory {frames_folder} is empty.")
        else:
            print(f"The directory {frames_folder} is not empty.")
        for filename in os.listdir(frames_folder):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(frames_folder, filename)
                frame = Image.open(frame_path)
                if route == "accurate":
                    try:
                        res_image, results = cycle_accurate(frame)
                        results_list.append(results)
                        frames_list.append(res_image)
                    except:
                        pass
                else:
                    try:
                        res_image, results = cycle_fast(frame)
                        results_list.append(results)
                        frames_list.append(res_image)
                    except:
                        pass
        shutil.rmtree(frames_folder)
        return results_list  

def count_classes(results):
    """
    Считает количество экземпляров каждого класса.
    """
    names_count = Counter(results[0].boxes.cls.tolist())
    return names_count

def ensemble_detect(models, frame, type):
    """
    Ансамблирование моделей и подсчет решения.
    """
    results_list = []
    if type == "seg":
        models_pack = [models[11],models[10],models[9]]

    if type == "det":
        models_pack = [models[8],models[7],models[6]]

    for model in models_pack:
            img, results = model(frame, verbose = False)
            results_list.append(results[0].probs.top1)

    

    
