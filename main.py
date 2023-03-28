from flask import Flask, render_template, redirect
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from data import db_session
from data.users import User
from forms.register import RegisterForm
from forms.login import LoginForm
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yandexlyceum_secret_key'
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


@app.route('/', methods=["GET"])
def main_screen():
    return render_template("main.html", title="Главный экран")


def main():
    db_session.global_init("db/database.db")
    app.run()


if __name__ == '__main__':
    main()
