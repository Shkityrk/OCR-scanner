import numpy as np
import h5py
from keras.models import load_model
from keras.utils import to_categorical


# Загрузка данных
def load_data(filename):
    with h5py.File(filename, 'r') as hf:
        X_test = np.array(hf.get('X_test'))
        y_test = np.array(hf.get('y_test'))
    return X_test, y_test


# Загрузка данных и модели
X_test, y_test = load_data('emnist_letters_old.h5')
model = load_model('emnist_letters_old.h5')  # Замените 'your_trained_model.h5' на вашу модель

# Преобразование меток в one-hot кодировку
y_test_one_hot = to_categorical(y_test)

# Оценка точности модели на тестовом наборе данных
accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
print("Accuracy: %.2f%%" % (accuracy[1] * 100))
