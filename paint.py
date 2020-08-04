# Importando o pacote de bibliotecas
import keras
from keras.models import load_model
import PIL
from PIL import Image, ImageGrab
import numpy as np
from tkinter import *
import tkinter as tk

model = load_model("model.pb")


def predict(image):
    # Redimensionando a imagem para 28x28 pixels
    image = image.resize((28,28))
    # Converter a imagem rgb em escala de cinza
    image = image.convert('L')
    # Converter a imagem em array matrix numpy
    image = np.invert(np.array(image))
    # Dá uma nova forma a uma matriz sem alterar seus dados.
    image = image.reshape(1,28,28)
    # Fazendo a previsão do numero
    prediction = model.predict(image)[0]
    return np.argmax(prediction)

# Criando a interface com tkinter
class App(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        # Criando elementos da interface
        self.canvas = tk.Canvas(self,width=500,height=500,bg='white',cursor='cross')
        self.label1 = tk.Label(self,text = 'Desenhe um número aqui!',font=('Ubuntu',50))
        self.label2 = tk.Label(self,font=('Ubuntu',50))
        self.frame = tk.Frame(self,width=500)
        clear_button = tk.Button(self.frame,text="Limpar",font=('Ubuntu',15),activebackground='green',command=self.clear)
        digit_button = tk.Button(self.frame,text="Testar",font=('Ubuntu',15),activebackground='green',command=self.digit)

        # Criando a istrutura do grid
        self.canvas.grid(row=1,column=0,pady=2,padx=2,sticky=W)
        self.label1.grid(row=0,column=0,pady=2,padx=2,sticky=W)
        self.label2.grid(row=3,column=0,pady=2,padx=2,sticky=W)
        self.frame.grid(row=2,column=0,pady=2,padx=2)

        clear_button.pack(side="left")
        digit_button.pack(side="right")

        self.canvas.bind("<B1-Motion>",self.draw)

    def clear(self):
        self.canvas.delete("all")
        self.label2.configure(text = "")

    # Obtendo o desenho do numero
    def digit(self):
        a = self.winfo_rootx() + self.canvas.winfo_x()
        b = self.winfo_rooty() + self.canvas.winfo_y()
        c = a + self.canvas.winfo_width()
        d = b + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((a,b,c,d))

        # Chamando a função predict
        predicted_digit = predict(image)

        # Imprimindo os valores na tela
        self.label2.configure(text = "Dígito Previsto: {}".format(str(predicted_digit)))

    def draw(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')


myapp = App()
myapp.title("Reconhecendo Digito com Uma Rede Neural")
myapp.mainloop()
