from flask import Flask, render_template, redirect, request, flash, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from data import db_session
from data.users import User
from forms.register import RegisterForm
from forms.login import LoginForm
import os
from werkzeug.utils import secure_filename
import sqlite3
import cv2
import numpy as np
from art import tprint

app = Flask(__name__)
con = sqlite3.connect("db/database.db")
cur = con.cursor()
tema = "car"
app.config['SECRET_KEY'] = 'yandexlyceum_secret_key'
# папка для сохранения загруженных файлов
# расширения файлов, которые разрешено загружать
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return db_session.create_session().query(User).get(user_id)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("/")


@app.route("/profile", methods=["GET"])
@login_required
def profile():
    return render_template("profile.html", title=f'Профиль {current_user.nickname}')


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        sess = db_session.create_session()
        user = sess.query(User).filter(User.email == form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect("/")
        return render_template("login.html", title='Авторизация', form=form,
                               message="Неверный логин или пароль")
    return render_template("login.html", title="Авторизация", form=form)


@app.route('/register', methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if form.password.data != form.password_again.data:
            return render_template('register.html', title='Регистрация',
                                   form=form, message='Пароли не совпадают')
        sess = db_session.create_session()
        if sess.query(User).filter(User.nickname == form.nickname.data).first():
            return render_template('register.html', title='Регистрация',
                                   form=form, message='Пользователь с данной почтой уже есть')
        f = form.picture.data
        f.save(os.path.join(os.getcwd(), f"static/img/avatars/{len(sess.query(User).all()) + 1}.jpg"))
        user = User(
            nickname=form.nickname.data,
            email=form.email.data,
            picture=f"{len(sess.query(User).all()) + 1}.jpg"
        )
        user.set_password(form.password.data)
        sess.add(user)
        sess.commit()
        return redirect('/login')
    return render_template('register.html', title='Регистрация', form=form)


@app.route("/get_img", methods=["GET", "POST"])
def get_img():
    if request.method == 'POST':
        # проверим, передается ли в запросе файл
        if 'file' not in request.files:
            # После перенаправления на страницу загрузки
            # покажем сообщение пользователю
            flash('Не могу прочитать файл')
            return redirect(request.url)
        file = request.files['file']
        # Если файл не выбран, то браузер может
        # отправить пустой файл без имени.
        if file.filename == '':
            flash('Нет выбранного файла')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # безопасно извлекаем оригинальное имя файла
            filename = secure_filename(file.filename)
            # сохраняем файл
            file.save(os.path.join(os.getcwd(), f"uploads_photo/test.jpg"))
            '''
            con = sqlite3.connect("db/database.db")
            cur = con.cursor()
            result = cur.execute(
                f"""INSERT INTO score(score, user_name) VALUES({radar_for_olega("uploads_photo/test.jpg", tema)},
        '{current_user.nickname}')""").fetchall()
            con.close()
            '''
            #TODO Вот эту херь надо заставить работать
            # если все прошло успешно, то перенаправляем
            # на функцию-представление `download_file`
            # для скачивания файла
            return redirect("/profile")
    return render_template("get_img.html", tema=tema)


@app.route('/', methods=["GET"])
def main_screen():
    return render_template("main.html", title="Главный экран")


def main():
    db_session.global_init("db/database.db")
    app.run()


def allowed_file(filename):
    """ Функция проверки расширения файла """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    main()
