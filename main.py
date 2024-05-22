from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from collections import Counter
import colorsys


def segmentation(image: Image, preprocessor, model) -> list[list[int]]:
    '''Возвращает список пикселей для каждого элемента одежды c помощью модели с Hugging Face'''
    image_matrix = np.array(image)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].tolist()

    colors = [[] for _ in range(18)]
    for i in range(len(pred_seg)):
        for j in range(len(pred_seg[0])):
            colors[int(pred_seg[i][j])].append(image_matrix[i][j])
    return colors


def find_common_color(array: list[list[int]]) -> list:
    '''Возвращает самый часто встречающийся пиксель в списке с альфа-каналом'''
    sublist, _ = max(Counter(tuple(x) for x in array).items(), key=lambda x: x[1])
    return list(sublist) + [255]


def get_colors_for_hair(hair_pixels: list[list[int]]):
    ''' Возвращает новый цвет волос и блеска (один из трех вариантов) 
        в зависимости от "светлости" волос на исходном изображении '''
    red, green, blue, _ = find_common_color(hair_pixels)
    # hair_metric- метрика "светлости" волос
    hair_metric = 1 - (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    if hair_metric <= 0.5:
        hair_color_to_change = [210, 168, 124, 255]
        hair_glare_color_to_change = [223, 184, 144, 255]
    elif 0.5 < hair_metric <= 0.79:
        hair_color_to_change = [161, 131, 100, 255]
        hair_glare_color_to_change = [210, 168, 124, 255]
    else:
        hair_color_to_change = [0, 0, 0, 255]
        hair_glare_color_to_change = [23, 23, 23, 255]
    return hair_color_to_change, hair_glare_color_to_change


def get_shadows(color: list[int]):
    '''Возвращает два цвета (две степени тени: светлую и темную)'''
    r, g, b, a = color
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    r_1, g_1, b_1 = colorsys.hsv_to_rgb(h, s, v - 0.15)
    r_2, g_2, b_2 = colorsys.hsv_to_rgb(h, s, v - 0.25)
    shadow_1 = np.array([r_1 * 255, g_1 * 255, b_1 * 255, a])
    shadow_2 = np.array([r_2 * 255, g_2 * 255, b_2 * 255, a])
    return shadow_1, shadow_2


def change_colors(image_tatarin: Image, dict_for_change_color: dict) -> Image:
    '''Возвращает изображение татарина с новыми цветами'''
    matrix = np.array(image_tatarin)
    for file_name, new_color in dict_for_change_colors.items():
        txt_file = open(f"indices/{file_name}", "r", encoding="utf-8")
        for row in txt_file:
            i, j = map(lambda x: int(x), row.split(" "))
            matrix[i][j] = new_color
        txt_file.close()
    result_image = Image.fromarray(matrix).convert('RGBA')
    return result_image


# Загружаем изображения
image = Image.open("images/example.jpg")
tatarin = Image.open("images/tatarin.png")

# Загружаем препроцессор и модель с Hugging Face
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Получаем пиксели волос, верхней одежды, штанов и обуви с изображения
colors = segmentation(image, processor, model)
hair_pixels, upper_clothes_pixels, pants_pixels, shoes_pixels = colors[2], colors[4], colors[6], colors[9]

# Получаем новые цвета для заливки нашего татарина
hair_color_to_change, hair_glare_color_to_change = get_colors_for_hair(hair_pixels)
upper_clothes_color_to_change = find_common_color(upper_clothes_pixels)
pants_color_to_change = find_common_color(pants_pixels)
shoes_color_to_change = find_common_color(shoes_pixels)
upper_clothes_shadow1_color_to_change, upper_clothes_shadow2_color_to_change = get_shadows(
    upper_clothes_color_to_change)

# Создаем словарь, где ключи- это названия файлов с индексами нужных пикселей, а значения- это новые цвета
dict_for_change_colors = {"hair_indices.txt": hair_color_to_change,
                          "hair_glare_indices.txt": hair_glare_color_to_change,
                          "upper_clothes_indices.txt": upper_clothes_color_to_change,
                          "upper_clothes_shadow1_indices.txt": upper_clothes_shadow1_color_to_change,
                          "upper_clothes_shadow2_indices.txt": upper_clothes_shadow2_color_to_change,
                          "pants_indices.txt": pants_color_to_change,
                          "shoes_indices.txt": shoes_color_to_change}

# Меняем цвет
new_tatarin = change_colors(tatarin, dict_for_change_colors)

# Сохраняем новое изображение
new_tatarin.save("images/new_tatarin.png")