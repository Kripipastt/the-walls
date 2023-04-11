# Импортируем необходимые классы.
import logging
from telegram.ext import Application, MessageHandler, filters, CommandHandler
from login import login_bot as login
from text_for_bot import privet, startuem
# Запускаем логгирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG
)

logger = logging.getLogger(__name__)


# Определяем функцию-обработчик сообщений.
# У неё два параметра, updater, принявший сообщение и контекст - дополнительная информация о сообщении.

def loggin(log, pasword):
    if log == "admin" and pasword == "admin":
        return True
    return False


async def person_dialog(update, context):
    hi = True
    login = True
    while True:
        print(update.message.text)
        if hi:
            await update.message.reply_text(privet)
            hi = False
        elif login:
            await update.message.reply_text('У тебя есть аккаунт?')
            login = False
    # У объекта класса Updater есть поле message,
    # являющееся объектом сообщения.
    # У message есть поле text, содержащее текст полученного сообщения,
    # а также метод reply_text(str),
    # отсылающий ответ пользователю, от которого получено сообщение.

async def help(update, context):
    await update.message.reply_text(privet)


async def start(update, context):
    await update.message.reply_text(startuem)

def main():
    # Создаём объект Application.
    # Вместо слова "TOKEN" надо разместить полученный от @BotFather токен
    application = Application.builder().token(login).build()

    # Создаём обработчик сообщений типа filters.TEXT
    # из описанной выше асинхронной функции echo()
    # После регистрации обработчика в приложении
    # эта асинхронная функция будет вызываться при получении сообщения
    # с типом "текст", т. е. текстовых сообщений.
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("start", start))

    # Запускаем приложение.
    application.run_polling()


# Запускаем функцию main() в случае запуска скрипта.
if __name__ == '__main__':
    main()
