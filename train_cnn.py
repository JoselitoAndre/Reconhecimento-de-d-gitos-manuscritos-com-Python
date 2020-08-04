import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

#carregar conjunto de dados
def load_dataset():
    # Divididos o cununto de dados entre treino e teste.
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    return x_train,y_train,x_test,y_test


# Pré-processar e normalizar dados
def preprocess(x_train,y_train,x_test,y_test):
    x_train = keras.utils.normalize(x_train,axis=1)
    x_test = keras.utils.normalize(x_test,axis=1)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return x_train,y_train,x_test,y_test


    # Ajustando o modelo
def define_model(input_shape,num_classes):

    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=60,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=num_classes,activation='softmax'))

    # Copilando model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# main
def main():
    # Carregando dados de teste
    x_train,y_train,x_test,y_test = load_dataset()
    # preprocess
    x_train,y_train,x_test,y_test = preprocess(x_train,y_train,x_test,y_test)
    print("Numero da amostras de treinamento: {}".format(x_train.shape[0]))
    print("Numero da amostras de teste : {}".format(x_test.shape[0]))
    # Definindo o modelo
    input_shape = (28,28)
    num_classes = 10
    model = define_model(input_shape,num_classes)
    # Treiando o Modelo
    history = model.fit(x_train,y_train,epochs=4)
    print("Treiando o Modelo")
    # Avaliando o modelo
    loss, accuracy = model.evaluate(x_test,y_test)
    print("loss : {}\naccuracy : {}".format(loss,accuracy))
    # Salvando o modelo para usar na gui
    model.save("model.pb")


if __name__ == "__main__":
    main()
