import time

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.dispatcher.filters import Text
from login import *
import cv2
import numpy as np
from art import tprint


bot = Bot(token=login_bot)
dp = Dispatcher(bot)
cod = '2510949905095343482370426320625501648248386831985'
tema = "car"

@dp.message_handler(commands=['start'])
async def starter(message: types.Message):
    await message.reply(startuem)

@dp.message_handler(commands=['tema'])
async def tema(message: types.Message):
    await message.reply("Тема сегоднешнего дня: " + tema)


@dp.message_handler(Text(equals="Старт"))
async def with_puree(message: types.Message):
    await message.reply(startuem)


@dp.message_handler(Text(equals="Помощь"))
async def with_puree_2(message: types.Message):
    await message.reply(privet)

@dp.message_handler(Text(equals="Тема"))
async def with_puree_3(message: types.Message):
    await message.reply("Тема сегоднешнего дня: " + tema)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply(privet)


@dp.message_handler()
async def start_bot(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup()
    button_1 = types.KeyboardButton(text="Старт")
    keyboard.add(button_1)
    button_2 = "Помощь"
    keyboard.add(button_2)
    button_3 = "Тема"
    keyboard.add(button_3)
    await message.reply(startuem, reply_markup=keyboard)


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def photo(message: types.Message):
    tema = "car"
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    photo = await file.download()
    time.sleep(3)
    print(file_id, file)
#    radar_for_olega(file["file_path"], tema)
    num_obgekt = "0"
    await bot.send_message(message.chat.id, radar_for_olega(file["file_path"], tema))



def apply_yolo_object_detection(image_to_process):
    """
    Recognition and determination of the coordinates of objects on the image
    :param image_to_process: original image
    :return: image with marked objects and captions to them
    """

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # Starting a search for objects in an image
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Selection
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # For debugging, we draw objects included in the desired classes
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image, objects_count = draw_object_count(image_to_process, objects_count)
    return final_image, objects_count


def draw_object_bounding_box(image_to_process, index, box):
    """
    Drawing object borders with captions
    :param image_to_process: original image
    :param index: index of object class defined with YOLO
    :param box: coordinates of the area around the object
    :return: image with marked objects
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font,
                              font_size, color, width, cv2.LINE_AA)

    return final_image


def draw_object_count(image_to_process, objects_count):
    """
    Signature of the number of found objects in the image
    :param image_to_process: original image
    :param objects_count: the number of objects of the desired class
    :return: image with labeled number of found objects
    """

    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    # Text output with a stroke
    # (so that it can be seen in different lighting conditions of the picture)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    return final_image, objects_count


def start_image_object_detection(img_path):
    """
    Image analysis
    """

    try:
        # Applying Object Recognition Techniques in an Image by YOLO
        image = cv2.imread(img_path)
        image, objects_count = apply_yolo_object_detection(image)
        return objects_count
        # return image
        # Displaying the processed image on the screen
        '''cv2.imshow("Image", image)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()'''

    except KeyboardInterrupt:
        pass


def radar_for_olega(img, tema):
    # Loading YOLO scales from files and setting up the network
    global net, classes, classes_to_look_for, out_layers
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg",
                                     "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Loading from a file of object classes that YOLO can detect
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    # Determining classes that will be prioritized for search in an image
    # The names are in the file coco.names.txt

    image = img
    look_for = tema.split(",")

    # Delete spaces
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    return start_image_object_detection(image)


if __name__ == '__main__':

    executor.start_polling(dp)
