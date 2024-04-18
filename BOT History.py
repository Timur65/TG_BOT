# Установка необходимых для работы программы модулей
# !pip install pytelegrambotapi -q
# !pip install g4f -q
# !pip install diffusers -q
# !pip install deep-translator -q


#Подключаем бота
import telebot;
bot = telebot.TeleBot('7140368368:AAGPBu3w6v6ZLAZiauYsBHFve8ya2A-1AAU'); #Вводим наш токен(бота)

# Наш мозг - или же основная мощь программы искуственный интелект на основе GPT4

import g4f
from g4f.Provider import Bing, OpenaiChat, Liaobots, BaseProvider
from g4f.cookies import set_cookies
from g4f.client import Client
from deep_translator import GoogleTranslator
import nest_asyncio

nest_asyncio.apply()

client= Client()
chat_history = [{"role": "user", "content": ''}]  # Сбор информации для ИИ(История сообщений)


def send_request(message):         # Сообщение генерируется на базе искуственного интелекта
    global chat_history            # Основываясь на вашем запросе, а также истории самих запросов
    chat_history[0]["content"] += message + " "
    print(chat_history)
    try:
        response = g4f.ChatCompletion.create(
        model=g4f.models.default,
        provider=g4f.Provider.OpenaiChat,
        messages=chat_history
    )
    except Exception as err:
        #time.sleep(120)
        response = g4f.ChatCompletion.create(
        model=g4f.models.default,
        provider=g4f.Provider.OpenaiChat,
        messages=chat_history
    )
    return (response)
    chat_history[0]["content"] += response + " "


# Наши кисточки - или же модули генерирующие картинки
# Первая имеет более сложную систему создания картинки(за счет "рафинирования"
from diffusers import DiffusionPipeline
import torch


# Первая кисточка
def send_photo(message):
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16", )
    refiner.to("cuda")


    n_steps = 40
    high_noise_frac = 0.8

    prompt = message

    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent", ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image, ).images[0]
    return image


# Вторая кисточка (Образец который я использую, с целью экономии времени в тестаъ)
def send_photo1(message):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                             use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    prompt = message

    images = pipe(prompt=prompt).images[0]
    return images


# Запуск бота, с зацикливанием, для постоянной работы
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Привет! Можешь спрашивать меня! Что тебя интересует?')


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    inp = GoogleTranslator(source='auto', target='en').translate(message.text)
    inp1 = "Tell us in detail about " + inp
    out = GoogleTranslator(source='auto', target='ru').translate(send_request(inp1))

    # Отправка текста
    bot.send_message(message.chat.id, out)
    # Первая кисточка - посложнее
    # bot.send_photo(message.chat.id, send_photo(inp))
    # Вторая кисточка - попроще
    bot.send_photo(message.chat.id, send_photo1(inp))


bot.polling(none_stop=True, interval=0)