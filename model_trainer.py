import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Импортируем Dropout
from data_loader import train_data_gen, val_data_gen, test_data_gen
import matplotlib.pyplot as plt
import numpy as np


# Определение архитектуры модели с добавлением Dropout
def create_model():
    model = Sequential([
        # Первый сверточный слой
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),

        # Добавляем слой Dropout
        Dropout(0.25),  # Отключаем 25% нейронов

        # Второй сверточный слой
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Добавляем слой Dropout
        Dropout(0.25),  # Отключаем 25% нейронов

        # Третий сверточный слой
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Добавляем слой Dropout
        Dropout(0.25),  # Отключаем 25% нейронов

        # Выпрямление данных
        Flatten(),

        # Полносвязные слои
        Dense(512, activation='relu'),

        # Добавляем слой Dropout
        Dropout(0.5),  # Отключаем 50% нейронов

        Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Построение модели
model = create_model()
model.summary()

# Обучение модели
EPOCHS = 20  # Увеличим количество эпох, чтобы увидеть эффект Dropout
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // val_data_gen.batch_size
)

# Визуализация результатов обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Оценка модели на тестовой выборке
test_loss, test_acc = model.evaluate(test_data_gen)
print(f'\nТочность на тестовой выборке: {test_acc:.4f}')

# Сохранение улучшенной модели
model.save('pneumonia_detection_model_v2.h5')
print('Улучшенная модель сохранена как pneumonia_detection_model_v2.h5')