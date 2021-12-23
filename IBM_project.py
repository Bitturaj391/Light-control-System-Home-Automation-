from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly as py
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as py
import plotly.offline

import datetime as date

pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)


class face:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1350x700+0+0")
        self.root.title("Project For IBM Internship".center(420))
        self.root.config(bg="Pink")
        # ------------Titles--------------------------------
        title = Label(self.root, text="Welcome to the Team Aryabhatta", font=(
            "times new roman", 30, "bold"), bg="#010c48", fg="white").place(x=0, y=0, relwidth=1, height=50)
        title2 = Label(self.root, text="Save Energy, Save Earth", font=("times new roman", 20),
                       bg="#010c48", fg="white").place(x=0, y=70, relwidth=1, height=50)
        # ----------------exit buton----------------------------------------------------------------
        btn_exit = Button(self.root, text="Exit", font=("times new roman", 20, "bold"), fg="white",
                          command=root.quit, bg="Blue", cursor="hand2").place(x=1200, y=120, height=40, width=150)
        # --------------left menu----------------------------------------------------------------
        self.menuLogo = Image.open("logo1.jpeg")
        self.menuLogo = self.menuLogo.resize((200, 200), Image.ANTIALIAS)
        self.menuLogo = ImageTk.PhotoImage(self.menuLogo)

        leftmenu = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        leftmenu.place(x=0, y=120, width=200, height=500)
        labelmenulogo = Label(leftmenu, image=self.menuLogo)
        labelmenulogo.pack(side=TOP, fill=X)
        label_menu = Label(leftmenu, text="Features", font=(
            "times new roman", 20, "bold"), bg="#010c48", fg="white").pack(side=TOP, fill=X)

        face = Button(leftmenu, text="face Detection", command=self.face1, font=(
            "times new roman", 15), fg="white", bg="Blue", cursor="hand2").pack(side=TOP, fill=X)
        analysis = Button(leftmenu, text="Analysis", command=self.analysis, font=(
            "times new roman", 15), fg="white", bg="Blue", cursor="hand2").pack(side=TOP, fill=X)

        chat = Button(leftmenu, text="Chat....", command=self.chat, font=(
            "times new roman", 15), fg="white", bg="Blue", cursor="hand2").pack(side=TOP, fill=X)
        # ------------------------Background------------------------
        self.background = Label(self.root, text="Developed By Team Aryabhatta", font=(
            "times new Roman", 50, "bold"), bg="#2F4F4F", fg="white")
        self.background.place(x=280, y=180, width=1000, height=80)
# --------------first member image------------------------
        self.menuLogo1 = Image.open("bittu pic.jpg")
        self.menuLogo1 = self.menuLogo1.resize((200, 200), Image.ANTIALIAS)
        self.menuLogo1 = ImageTk.PhotoImage(self.menuLogo1)

        leftmenu = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        leftmenu.place(x=310, y=270, width=200, height=200)
        labelmenulogo1 = Label(leftmenu, image=self.menuLogo1)
        labelmenulogo1.pack(side=TOP, fill=X)

        bittu = Button(self.root, text="Bittu Raj", font=("times new roman", 20, "bold"), fg="white",
                        bg="gray").place(x=310, y=470, height=40, width=200)
# --------2ND MEMEBER IMAGE-----------------------

        self.menuLogo2 = Image.open("shankar.jpg")
        self.menuLogo2 = self.menuLogo2.resize((200, 200), Image.ANTIALIAS)
        self.menuLogo2 = ImageTk.PhotoImage(self.menuLogo2)

        leftmenu = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        leftmenu.place(x=550, y=270, width=200, height=200)
        labelmenulogo2 = Label(leftmenu, image=self.menuLogo2)
        labelmenulogo2.pack(side=TOP, fill=X)

        shankar = Button(self.root, text="Shankar Kumar", font=("times new roman", 20, "bold"), fg="white",
                     bg="gray").place(x=550, y=470, height=40, width=200)
# --------------------3RD MEMBER IMAGE----------------

        self.menuLogo3 = Image.open("devraj.jpg")
        self.menuLogo3 = self.menuLogo3.resize((200, 200), Image.ANTIALIAS)
        self.menuLogo3 = ImageTk.PhotoImage(self.menuLogo3)

        leftmenu = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        leftmenu.place(x=790, y=270, width=200, height=200)
        labelmenulogo3 = Label(leftmenu, image=self.menuLogo3)
        labelmenulogo3.pack(side=TOP, fill=X)

        devraj = Button(self.root, text="Devraj", font=("times new roman", 20, "bold"), fg="white", bg="gray").place(x=790, y=470, height=40, width=200)
# ------------------4th member Image-------------------
        self.menuLogo4 = Image.open("riyaz.jpeg")
        self.menuLogo4 = self.menuLogo4.resize((200, 200), Image.ANTIALIAS)
        self.menuLogo4 = ImageTk.PhotoImage(self.menuLogo4)

        leftmenu = Frame(self.root, bd=2, relief=RIDGE, bg="white")
        leftmenu.place(x=1030, y=270, width=200, height=200)
        labelmenulogo4 = Label(leftmenu, image=self.menuLogo4)
        labelmenulogo4.pack(side=TOP, fill=X)

        Riyaz = Button(self.root, text="Riyaz Ansari", font=("times new roman", 20, "bold"), fg="white", bg="gray").place(x=1030, y=470, height=40, width=200)

        # -----------face detection -----------------------------------
    def face1(self):
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

        # To capture video from webcam.
        cap = cv2.VideoCapture(0)
        # To use a video file as input
        # cap = cv2.VideoCapture('filename.mp4')

        while True:
            # Read the frame
            _, img = cap.read()

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display
            cv2.imshow('img', img)

            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            if faces.any():
                print("face is detected")
            
        # Release the VideoCapture object
        cap.release()

    def analysis(self):
        data = pd.read_csv('dataset_tk.csv')
        long = pd.read_csv('long_data_.csv')
        data_copy = data.copy()
        data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

        data['date'] = pd.to_datetime(data['date'])
        long['Dates'] = pd.to_datetime(long['Dates'])
        data_copy = data.copy()

        data.set_index('date', inplace=True)
        data['Total'] = data.sum(axis=1)
        data.loc['Total'] = data.sum()
        Total_comsumes = data.iloc[[-1]]
        Total = Total_comsumes.transpose()
        Total.drop(Total.index[[-1]], inplace=True)
        Total.reset_index(inplace=True)
        Total.rename(columns={'index': 'State'}, inplace=True)
        fig = go.Figure(go.Pie(labels=Total['State'],
                               values=Total['Total'], title='State Wise Electricity Consumption'))
        iplot(fig)
        Top_10_EC_Consume = Total.nlargest(10, 'Total')
        print(Top_10_EC_Consume)
        layout = go.Layout(title="Maximum power consume state", xaxis={
                           'title': 'EC'}, yaxis={'title': 'State'})
        cons = go.Bar(
            x=Top_10_EC_Consume['Total'], y=Top_10_EC_Consume['State'], orientation='h')
        fig = go.Figure(data=cons, layout=layout)
        iplot(fig)
        Low_EC_Consumption = Total.nsmallest(10, 'Total')
        print(Low_EC_Consumption)
        layout = go.Layout(title="Minimum power consume state", xaxis={
                           'title': 'EC'}, yaxis={'title': 'State'})
        d = go.Bar(x=Low_EC_Consumption['Total'],
                   y=Low_EC_Consumption['State'], orientation='h')
        fig = go.Figure(data=d, layout=layout)
        iplot(fig)
        grp = long.groupby('Regions')
        re = grp.agg(np.size)
        regine = re[['Usage']]
        regine.reset_index(inplace=True)
        fig = go.Figure(go.Pie(
            labels=regine['Regions'], values=regine['Usage'], title='Regine wise Electricity Consumption'))
        iplot(fig)
        data_copy.set_index('date', inplace=True)
        data_resample = data_copy.resample('MS').sum()
        data_resample.reset_index(inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Punjab'],
                                 mode='lines',
                                 name='Punjab'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Haryana'],
                                 mode='lines',
                                 name='Haryana'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Rajasthan'],
                                 mode='lines',
                                 name='Rajasthan'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Delhi'],
                                 mode='lines',
                                 name='Delhi'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['UP'],
                                 mode='lines',
                                 name='UP'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Uttarakhand'],
                                 mode='lines',
                                 name='Uttarakhand'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['HP'],
                                 mode='lines',
                                 name='HP'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['J&K'],
                                 mode='lines',
                                 name='J&K'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Chandigarh'],
                                 mode='lines',
                                 name='Chandigarh'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Chhattisgarh'],
                                 mode='lines',
                                 name='Chhattisgarh'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Gujarat'],
                                 mode='lines',
                                 name='Gujarat'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['MP'],
                                 mode='lines',
                                 name='MP'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Maharashtra'],
                                 mode='lines',
                                 name='Maharashtra'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Goa'],
                                 mode='lines',
                                 name='Goa'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['DNH'],
                                 mode='lines',
                                 name='DNH'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Andhra Pradesh'],
                                 mode='lines',
                                 name='Andhra Pradesh'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Telangana'],
                                 mode='lines',
                                 name='Telangana'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Karnataka'],
                                 mode='lines',
                                 name='Karnataka'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Kerala'],
                                 mode='lines',
                                 name='Kerala'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Tamil Nadu'],
                                 mode='lines',
                                 name='Tamil Nadu'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Pondy'],
                                 mode='lines',
                                 name='Pondy'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Bihar'],
                                 mode='lines',
                                 name='Bihar'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Jharkhand'],
                                 mode='lines',
                                 name='Jharkhand'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Odisha'],
                                 mode='lines',
                                 name='Odisha'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['West Bengal'],
                                 mode='lines',
                                 name='West Bengal'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Sikkim'],
                                 mode='lines',
                                 name='Sikkim'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Arunachal Pradesh'],
                                 mode='lines',
                                 name='Arunachal Pradesh'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Assam'],
                                 mode='lines',
                                 name='Assam'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Manipur'],
                                 mode='lines',
                                 name='Manipur'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Meghalaya'],
                                 mode='lines',
                                 name='Meghalaya'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Mizoram'],
                                 mode='lines',
                                 name='Mizoram'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Nagaland'],
                                 mode='lines',
                                 name='Nagaland'))
        fig.add_trace(go.Scatter(x=data_resample['date'], y=data_copy['Tripura'],
                                 mode='lines',
                                 name='Tripura'))
        fig.update_layout(title='Power Consumption in  states')
        iplot(fig)
    def chat(self):
        print("I will feel happy to help you.")
        name = input('Please enter your name.\n')
        problem = input("Please write your problem.\n")
        print("Thank you for contacting us",name,"we will reach you soon with a proper solution")

if __name__ == "__main__":
    root = Tk()
    object = face(root)
    root.mainloop()
