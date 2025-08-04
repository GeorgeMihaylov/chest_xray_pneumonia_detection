import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

data_dir = 'chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Параметры изображений
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Создание генераторов данных
# ImageDataGenerator автоматически выполняет масштабирование пикселей
# и аугментацию для тренировочных данных
train_image_generator = ImageDataGenerator(
    rescale=1./255, # Нормализация пикселей к диапазону [0, 1]
    rotation_range=15, # Аугментация: поворот изображения на 15 градусов
    width_shift_range=0.1, # Сдвиг по ширине
    height_shift_range=0.1, # Сдвиг по высоте
    shear_range=0.1, # Сдвиг по углу
    zoom_range=0.1, # Увеличение изображения
    horizontal_flip=True, # Отражение по горизонтали
    fill_mode='nearest' # Как заполнять новые пиксели
)

validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Загрузка данных из папок
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary' # Так как у нас два класса (NORMAL и PNEUMONIA)
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

# Проверка, что данные загружены
print("Данные для обучения загружены. Количество классов:", train_data_gen.num_classes)
print("Названия классов:", list(train_data_gen.class_indices.keys()))