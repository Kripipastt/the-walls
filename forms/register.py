from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField, EmailField, FileField
from wtforms.validators import DataRequired


class RegisterForm(FlaskForm):
    nickname = StringField('Nickname', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    password_again = PasswordField('Again password', validators=[DataRequired()])
    email = EmailField('Email', validators=[DataRequired()])
    picture = FileField('Select avatar', validators=[DataRequired()])
    submit = SubmitField("Register")
