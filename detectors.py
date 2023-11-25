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

def run_full_cycle(models, data, type = "video", video_params = [110, 130, 20], ensemble = False, route = "accurate"):
    """
    Функция представляет собой полный цикл обработки данных, будь то видео или изображение
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
        
    if type == "video":
       return full_video_cycle(models, data, type = "video", video_params = [110, 130, 20], ensemble = False, route = "accurate")
    
    if type == "zip":
        zip_results_list = []
        zip_folder = utils.unzip_file(data)
        for root, dirs, files in os.walk(zip_folder):
            for file in files:
                zip_results_list.append(os.path.join(root, file))
        print(zip_results_list)

    
def full_video_cycle(models, data, type = "video", video_params = [110, 130, 20], ensemble = False, route = "accurate"):
        frames_folder, fps = utils.video_to_frames(data, video_params[0], video_params[1], video_params[2])
        print("Сбор путей фреймов")
        frames_paths = []
        for root, dirs, files in os.walk(frames_folder):
            for file in files:
                frames_paths.append(os.path.join(root, file))
        print("Проверка на валидность")
        results_list = []
        for frame in frames_paths:
            img, results = run_model(models, frame, "class_correct")
            results_list.append(results[0].probs.top1)
        print("Добавление валидных путей")
        first_check_list = []
        for idx, result in enumerate(results_list):
            if result == 0:
                first_check_list.append(frames_paths[idx])
        print("Проверка качества фреймов")
        results_list = []
        for frame in first_check_list:
            img, results = run_model(models, frame, "class_truck")
            results_list.append(results[0].probs.top1)
        results_list
        print("Добавление качественных фреймов")
        second_check_list = []
        for idx, result in enumerate(results_list):
            if result == 1:
                second_check_list.append(frames_paths[idx])
        print("Загрузка изображений")
        images = []
        for frame in second_check_list:
            image = Image.open(frame)
            images.append(image)

        if route == "accurate":
            print('Старт "точного" пути')
            segmentations_list = []
            for image in images:
                image, results = run_model(models, image, "segment")
                segmentations_list.append(results)

            masks_list = []
            for result in segmentations_list:
                mask = get_mask(result)
                mask = Image.fromarray(mask)
                masks_list.append(mask)

            classifications = []
            for mask in masks_list:
                cls_seg_image, cls_seg_results = run_model(models, mask, "class_segment")
                classifications.append(cls_seg_results[0].probs.top1)
            segm_result = utils.most_common(results_list)

            print('Проверка запуска ансамбля')
            if ensemble == False:
                return segm_result
            else:
                ans_results_list = []
                for mask in masks_list:
                    result = ensemble_detect(models, mask, "segment")
                    ans_results_list.append(result)
                ans_results_list.append(segm_result)
                full_result = utils.most_common(ans_results_list)
                return full_result
            
        elif route == "fast":
            detections_list = []
            detection_results = []
            for image in images:
                image, results = run_model(models, image, "detect")
                detections_list.append(results)
                detection_results.append(results[0].boxes.cls[0].cpu().numpy().astype(int))
            main_detection_result = utils.most_common(detection_results)

            crops_list = []
            for result in detections_list:
                crop = get_crop(result)
                crop = Image.fromarray(crop)
                crops_list.append(crop)

            classifications = []
            for crop in crops_list:
                cls_det_image, cls_det_results = run_model(models, crop, "class_detect")
                classifications.append(cls_det_results[0].probs.top1)

            classifications.append(main_detection_result)
            det_result = utils.most_common(classifications)
            print('Проверка запуска ансамбля')
            if ensemble == False:
                return det_result
            else:
                ans_results_list = []
                for mask in masks_list:
                    result = ensemble_detect(models, mask, "detect")
                    ans_results_list.append(result)
                ans_results_list.append(det_result)
                full_result = utils.most_common(ans_results_list)
                return full_result

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
    if type == "segment":
            img, results1 = run_model(models, frame, "cls_seg_ansemble1")
            img, results2 = run_model(models, frame, "cls_seg_ansemble2")
            img, results3 = run_model(models, frame, "cls_seg_ansemble3")
            results_list = [results1[0].probs.top1, results2[0].probs.top1, results3[0].probs.top1]
            most_common_class = max(results_list, key = results_list.count)
            return most_common_class

    elif type == "detect":
            img, results1 = run_model(models, frame, "cls_det_ansemble1")
            img, results2 = run_model(models, frame, "cls_det_ansemble2")
            img, results3 = run_model(models, frame, "cls_det_ansemble3")
            results_list = [results1[0].probs.top1, results2[0].probs.top1, results3[0].probs.top1]
            most_common_class = max(results_list, key = results_list.count)
            return most_common_class
    else:
        print("Неверный тип!")
        return

    

    

    
