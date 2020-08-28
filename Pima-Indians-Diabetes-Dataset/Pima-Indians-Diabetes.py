from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.utils import shuffle
from numpy import loadtxt
import requests
import os.path


def train(X, y):
    model = Sequential()
    model.add(Dense(13, input_dim=8, activation='relu'))
    model.add(Dense(10, input_dim=13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=200, batch_size=30)

    _, accuracy = model.evaluate(X, y)
    #model.save('test')
    X, y = shuffle(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

def download_file(filename, src=r'https://raw.github.com/jbrownlee/Datasets/master/'):
    print ('Downloading ', filename)
    data = requests.get(src+filename)
    data = data.text

    file = open('pima-indians-diabetes.csv', 'w')
    file.write(data).close()

def load_data(filename):
    if not os.path.exists(filename):
        download_file(filename)

    return loadtxt(filename, delimiter=',')


dataset = load_data('pima-indians-diabetes.csv')
train_X, train_y = dataset[:,0:8], dataset[:,8]

train(train_X, train_y)
