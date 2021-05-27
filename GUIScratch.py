import tkinter as tk
from PIL import Image, ImageDraw
from NeuralNetworkFromScratch import *


class Window:
    count = 0
    canvas = None
    label_status = None
    img = None
    img_draw = None
    win = None
    model = None

    def __init__(self):
        win = tk.Tk()
        self.scratchModel = self.loadNeuralNetwork()
        self.canvas = tk.Canvas(self.win, width=500, height=500, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4)

        button_save = tk.Button(win, text='SAVE', bg='green', fg='white', font='Helvetica 20 bold', command=self.save)
        button_save.grid(row=1, column=0)

        button_predict = tk.Button(win, text='PREDICT', bg='blue', fg='white', font='Helvetica 20 bold', command=self.predict)
        button_predict.grid(row=1, column=1)

        button_clear = tk.Button(win, text='CLEAR', bg='yellow', fg='white', font='Helvetica 20 bold', command=self.clear)
        button_clear.grid(row=1, column=2)

        button_exit = tk.Button(win, text='EXIT', bg='red', fg='white', font='Helvetica 20 bold', command=win.destroy)
        button_exit.grid(row=1, column=3)

        self.label_status = tk.Label(win, text='PREDICTED DIGIT: NONE', bg='white', font='Helvetica 18 bold')
        self.label_status.grid(row=2, column=0, columnspan=4)

        self.canvas.bind('<B1-Motion>', self.event_function)
        self.img = Image.new('RGB', (500, 500), (0, 0, 0))
        self.img_draw = ImageDraw.Draw(self.img)

        win.mainloop()

    def clear(self):
        self.canvas.delete('all')
        self.img = Image.new('RGB', (500, 500), (0, 0, 0))
        self.img_draw = ImageDraw.Draw(self.img)

        self.label_status.config(text='PREDICTED DIGIT: NONE')

    def event_function(self, event):
        x = event.x
        y = event.y

        # create an oval
        x1 = x - 25
        y1 = y - 25

        x2 = x + 25
        y2 = y + 25
        self.canvas.create_oval((x1, y1, x2, y2), fill='black')
        self.img_draw.ellipse((x1, y1, x2, y2), fill='white')

    def save(self):
        img_array = np.array(self.img)
        img_array = cv2.resize(img_array, (28, 28))

        cv2.imwrite(str(self.count) + ' .jpg', img_array)
        self.count += 1
        self.clear()
        self.label_status.config(text='DIGIT HAS BEEN SAVED')

    def predict(self):
        img_array = np.array(self.img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (28,28))

        img_array = img_array.astype('float32')
        img_array = img_array / 255.0
        img_array = img_array.reshape(784,)

        nn = self.scratchModel.predictModel(img_array)
        label = np.argmax(nn, axis=1)

        self.label_status.config(text='PREDICTED DIGIT: ' + str(label))

    def loadNeuralNetwork(self):
        scratchModel = NeuralNetworkFromScratch()

        return scratchModel