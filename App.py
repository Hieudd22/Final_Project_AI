from cProfile import label
from pickle import FROZENSET
from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
from PIL import  Image,ImageTk
from matplotlib.image import thumbnail
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np


model = load_model('C:/Users/ADMIN/Desktop/AI/Cuoi_ki/Data9-1/best_model.h5')

classes = ["Bach_dang_nk","Cao_su","Dang_huong_viet","Cho_chi","Anh_dao",
           "Hoang_dan","Tau_mat","Lim","Iroko","Mit_mat",
           "Muong_den","Lat_hoa","Sa_moc_dau","Trai_li","Thuy_tung"]

def showimage():
    global fln
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), 
                                     title="Select Image File", 
                                     filetypes=(("JPG File",".jpg"),
                                                ("PNG File",".png"), 
                                                ("ALL Files",".")))
    img = Image.open(fln)
    img.thumbnail((300,300))
    img = ImageTk.PhotoImage(img)
    lbl3.configure(image= img)
    lbl3.image = img

def recognize():
    global lbl1
    global lbl2
    img_path= fln
    img=plt.imread(img_path)
    print ('Input image shape is ', img.shape)
    img=cv2.resize(img, (150,150)) 
    print ('the resized image has shape ', img.shape)
    plt.axis('off')
    plt.imshow(img)
   
    img=np.expand_dims(img, axis=0)
    print ('image shape after expanding dimensions is ',img.shape)
    y_pred=model.predict(img)
    score = y_pred.max()
    score = round(score*100,2)
    y_classes = [np.argmax(element) for element in y_pred]
    print ('the shape of prediction is ', y_pred.shape)

    print(f'the image is predicted as being {classes[y_classes[0]]} with a probability of {score} %')
    lbl1 = Label(root,text = f"Nhận Diện: {classes[y_classes[0]]}" , fg= "green", font=("Arial", 20))
    lbl1.pack(pady= 20)
    lbl2 = Label(root,text = f"Độ chính xác: {score} %" , fg= "blue", font=("Arial", 20))
    lbl2.pack(pady= 20)
    return

def clear():
    lbl1.after(1000, lbl1.destroy())
    lbl2.after(1000, lbl2.destroy())
    return


root = Tk()
root.title("Wood Recognition")
root.geometry("600x600")


frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = Label(root,
            text = "Wood Recognition", 
            fg= "red", 
            font=("Arial", 30))
lbl.pack(padx=10, pady= 10)

lbl3 = Label(root)
lbl3.pack()

btn1 = Button(frm,text = "Browser Image", command= showimage)
btn1.pack(side=tk.LEFT,padx= 15)

btn2 = Button(frm,text = "Recognize", command= recognize)
btn2.pack(side=tk.LEFT,padx= 15, pady = 10)

btn3 = Button(frm,text = "Clear", command= clear )
btn3.pack(side=tk.LEFT,padx=20)

btn4 = Button(frm,text = "EXIT", command=lambda: exit())
btn4.pack(side=tk.LEFT,padx=20)


root.mainloop()