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

def run_model(models, image, type = "segment"):
    """
    Функция для детекции изображения одиночной моделью.
    """
    if type == "segment":
        print("Сегментирую")
        results = models[0](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_garbage":
        print("Классифицирую мусор")
        results = models[1](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_truck":
        print("Классифицирую наличие грузовика")
        results = models[2](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "detect":
        print("Детекчу")
        results = models[3](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results
    if type == "class_correct":
        print("Проверяю корректность изображения")
        results = models[4](image, verbose=False)
        im_array = results[0].plot()
        result_img = Image.fromarray(im_array[..., ::-1])
        return result_img, results

def run_full_cycle(models, data, type, ensemle = False, route = "accurate"):
    """
    Функция представляет собой полный цикл обработки данных, будь то видео или изображение

    Parameters:
    models: пачка моделей.
    data: данные.
    ensemble: используется ли ансамбль.
    route: выбор типа цикла 'fast' или 'accurate'.
    """
    def cycle_accurate(image, half = False):
        result_img, results = run_model(models, image, "class_truck")
        if half != True:
            if results[0].probs.top1 == 1 or results[0].probs.top1 == "1":
                result_img_seg, results_seg = run_model(models, image, "segment")
                result_img_mask = get_mask(results_seg)
                result_img_cls, results_cls = run_model(models, result_img_mask, "class_garbage")
                return result_img_cls, results_cls
            else:
                if results[0].probs.top1 == 1 or results[0].probs.top1 == "1":
                    result_img_seg, results_seg = run_model(models, image, "segment")
                    return result_img_seg, results_seg
            
    def cycle_fast(image, half = False):
        result_img, results = run_model(models, image, "class_truck")
        if half != True:
            if results[0].probs.top1 == 1 or results[0].probs.top1 == "1":
                result_img_det, results_det = run_model(models, image, "detect")
                result_img_mask = get_mask(results_det)
                result_img_cls, results_cls = run_model(models, result_img_mask, "class_garbage_detect")
                return result_img_cls, results_cls
            else:
                if results[0].probs.top1 == 1 or results[0].probs.top1 == "1":
                    result_img_det, results_det = run_model(models, image, "detect")
                    return result_img_det, results_det

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

def ensemble_detect(models_pack, image, time = None, type = "image"):
    """
    Ансамблирование моделей и подсчет решения.
    """
    try:
        image = image[...,::-1]
    except TypeError:
        print("Неверный формат массива")
    if type == "image":
        results_list = [count_classes(model(image)) for model in models_pack]

        most_common_values = Counter()
        for result in results_list:
            most_common_values += result

        output_list = most_common_values.most_common()
        result_img = run_model(models_pack[1], image)

        return result_img, output_list