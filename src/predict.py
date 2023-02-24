import numpy as np
import pandas as pd
import tkinter as tk
import customtkinter as ctk
import joblib
import config
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
from tkinter import filedialog
from tkinter.messagebox import showerror
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import process_image


class ImageViewer(ctk.CTkFrame):
    """
    Widget to display and navigate between images in the dataset
    """

    def __init__(self, master):
        # Initialize values
        self.button_setting = {
            'corner_radius': 10,
            'border_spacing': 6,
            'compound': 'top',
            'anchor': 's',
            'font': ctk.CTkFont(size=15)
        }
        super().__init__(master, corner_radius=0)
        self.master = master

        # Setting layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_columnconfigure(4, weight=1)

        # Images attributes
        self.images = []
        self.images_label = []
        self.index = 0

        # Add image label
        self.label = ctk.CTkLabel(self, text="", compound='top')
        self.label.grid(row=0, column=0, sticky='nsew', columnspan=5, padx=10, pady=10)

        # Add previous button
        self.previous_button = ctk.CTkButton(self, text="Previous", command=self.previous_image, **self.button_setting)
        self.previous_button.grid(row=1, column=1, sticky='ew', padx=10, pady=10)

        # Add next button
        self.next_button = ctk.CTkButton(self, text="Next", command=self.next_image, **self.button_setting)
        self.next_button.grid(row=1, column=3, sticky='ew', padx=10, pady=10)


    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()


    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_image()


    def show_image(self):
        # Display image and label
        image = ctk.CTkImage(Image.fromarray(self.images[self.index] * 255), size=(300, 300))
        label = self.images_label[self.index]
        self.label.configure(image=image, text=f'Image Label: {label}', font=("'Helvetica'", 20))



class PredictStartFrame(ctk.CTkFrame):
    """
    Starting Frame to let user choose between Live Prediction or Batch Prediction
    """

    def __init__(self, container):
        # Initialize values
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        self.button_setting = {
            'corner_radius': 10,
            'height': 120,
            'width': 170,
            'border_spacing': 10,
            'compound': 'top',
            'anchor': 's',
            'text_color': 'gray90',
            'hover_color': 'gray30',
            'fg_color': '#292929',
            'font': ctk.CTkFont(size=18, weight="bold")
        }

        # Add grid 3 x 5
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_columnconfigure(4, weight=2)

        # Add Live Prediction Start Button
        self.live_icon = ctk.CTkImage(Image.open(self.image_path + 'predict_live_icon.png'), size=(40, 40))
        self.live_button = ctk.CTkButton(self, image=self.live_icon, text='Live Prediction', 
                                         command=lambda: self.open_frame('live'), **self.button_setting)
        self.live_button.grid(row=1, column=1)

        # Add Batch Prediction Start Button
        self.batch_icon = ctk.CTkImage(Image.open(self.image_path + 'predict_batch_icon.png'), size=(40, 40))
        self.batch_button = ctk.CTkButton(self, image=self.batch_icon, text='Batch Prediction', 
                                          command=lambda: self.open_frame('batch'), **self.button_setting)
        self.batch_button.grid(row=1, column=3)


    def open_frame(self, name):
        if name == 'live':
            self.grid_forget()
            self.container.predict_live_frame.grid(row=0, column=1, sticky='nsew')
        else:
            result = filedialog.askdirectory(initialdir=config.USER_PATH) 
            if len(result) > 0:
                # Update dataset name and display the batch frame
                self.container.predict_batch_frame.dataset_name = result.split(r'/')[-1]
                self.container.predict_batch_frame.directory = result + r'//'
                self.grid_forget()
                self.container.predict_batch_frame.update_label()
                self.container.predict_batch_frame.grid(row=0, column=1, sticky='nsew')


    def close(self):
        self.grid_forget()


class PredictLivePredictionFrame(ctk.CTkFrame):
    """
    A Frame that let user draw and predict image.
    """

    def __init__(self, container):
        # Initialize values
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        self.button_setting = {
            'corner_radius': 10, 
            'compound': 'left',  
            'anchor': 'w', 
            'border_spacing': 10, 
            'height': 25,
            'text_color': 'gray90', 
            'hover_color': 'gray30', 
            'fg_color': '#3d85c6',
            'font': ctk.CTkFont(size=18, weight="bold")
        }

        # Add 2x2 grid
        self.grid_columnconfigure(0, weight=4)
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=2)

        # Add canvas
        self.canvas = ctk.CTkCanvas(self, bg='#bcbcbc', height=config.CANVAS_HEIGHT, width=config.CANVAS_WIDTH)
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.canvas_width = self.canvas.winfo_screenwidth()
        self.canvas_height = self.canvas.winfo_screenheight()

        # Image backend
        self.current_image = Image.new('L', (self.canvas_width, self.canvas_height))
        self.draw = ImageDraw.Draw(self.current_image)

        # Add clear button
        self.clear_button = ctk.CTkButton(self.canvas, text='clear',  width=30, corner_radius=0, command=self.clear_canvas)
        self.clear_button.place(x=0, y=0)

        # Add button frame and buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=0, column=1, sticky='nsew')
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        self.button_frame.grid_rowconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(3, weight=1)
        self.button_frame.grid_rowconfigure(4, weight=1)
        self.button_frame.grid_rowconfigure(5, weight=1)

        # Add model label
        self.ml_icon = ctk.CTkImage(Image.open(self.image_path + 'ml_icon.png'), size=(80, 80))
        self.model_label = ctk.CTkLabel(self.button_frame, font=ctk.CTkFont(size=20, weight="bold"), compound='top', 
                                        image=self.ml_icon, height=60, text='Live Prediction')
        self.model_label.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Add draw button
        self.paint_icon = ctk.CTkImage(Image.open(self.image_path + 'paint_icon.png'), size=(25, 25))
        self.draw_button = ctk.CTkButton(self.button_frame, image=self.paint_icon, text='Draw', 
                                         command=self.draw_button_event, **self.button_setting)
        self.draw_button.grid(row=1, column=0, padx=10, pady=15, sticky='ew')

        # Add predict button
        self.model_image_icon = ctk.CTkImage(Image.open(self.image_path + 'ai_icon.png'), size=(30, 30))
        self.predict_button = ctk.CTkButton(self.button_frame, image=self.model_image_icon, text='Predict', 
                                            **self.button_setting, command=self.predict_button_event)
        self.predict_button.grid(row=2, column=0, padx=10, pady=15, sticky='ew')

        # Adjust thick
        self.pen_width = 20
        self.thick_combobox = ctk.CTkComboBox(self.button_frame, values=['Pen Thickness: 10', 'Pen Thickness: 20', 'Pen Thickness: 30', 'Pen Thickness: 40'], 
                                              command=self.update_pen_width, height=30)
        self.thick_combobox.set('Choose Pen Thickness')
        self.thick_combobox.grid(row=3, column=0, padx=10, pady=15, sticky='ew')

        # Add choose model
        self.model_menu = ctk.CTkComboBox(self.button_frame, height=30, values=config.MODEL_LIST, command=self.model_menu_event)
        self.model_menu.set('Choose Algorithm')
        self.model_menu.grid(row=4, column=0, padx=10, pady=15, sticky='ew')
        
        """
        # Add model_name_label
        self.model_name_label = ctk.CTkLabel(self.button_frame, text='Model name: ', font=ctk.CTkFont(size=20, weight="bold"))
        self.model_name_label.grid(row=5, column=0, padx=10, pady=15, sticky='ew')
        """

        # Add message box
        self.message_box = ctk.CTkTextbox(self, font=ctk.CTkFont(size=15))
        self.message_box.insert(tk.INSERT, 'MESSAGE BOX\n')
        self.message_box.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.message_box.configure(state="disabled")

        # Model backend
        self.model_type = None
        self.model = None
        self.model_name = ''


    def clear_canvas(self):
        self.canvas.delete('all')
        # New image backend
        self.current_image = Image.new('L', (self.canvas_width, self.canvas_height))
        self.draw = ImageDraw.Draw(self.current_image)


    def draw_button_event(self):
        self.draw_button.configure(fg_color='gray25')
        self.canvas.configure(cursor='pencil')
        self.canvas.bind('<1>', self.activate_paint)


    def activate_paint(self, event):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.lastx, self.lasty = event.x, event.y


    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_line((self.lastx, self.lasty, x, y), width=self.pen_width, fill='black')
        self.draw.line([self.lastx, self.lasty, x, y], 255, width=self.pen_width)
        self.lastx, self.lasty = x, y


    def update_pen_width(self, choice):
        self.pen_width = int(choice.split()[-1])

    
    def model_menu_event(self, choice):
        self.model_type = choice
        
        # Open model in the dir
        result = filedialog.askopenfilename(initialdir=config.MODEL_PATH + self.model_type)
        # filedialog.askdirectory(initialdir=config.MODEL_PATH + self.model_type)
        if len(result) == 0:
            showerror(message="Please choose a model", title='Error')
        else:
            # Load the model
            if self.model_type == 'CNN':
                self.model = load_model(result)
            else:
                self.model = joblib.load(result)


    def predict_button_event(self):
        # Check choose model
        if self.model is None:
            showerror(message="Please choose a model first", title='Error')
        else:
            image_path = config.CURRENT_IMAGE_PATH
            self.current_image.save(image_path)
            img_array = process_image(image_path)
            if self.model_type != 'CNN':
                data = img_array.reshape((1, -1))
                digit = self.model.predict(data)
                digit = digit[0]
            else:
                digit = self.model.predict(x=img_array.reshape((1, 28, 28, 1))).argmax()
            
            # Print out the result
            self.message_box.configure(state="normal")
            self.message_box.insert(tk.END, f'\n- Your image predicted label is: {digit}')
            self.message_box.configure(state="disabled")

    def close(self):
        # Clear the message box and canvas
        self.message_box.configure(state="normal")
        self.message_box.delete('2.0', 'end')
        self.message_box.configure(state="disabled")
        self.clear_canvas()

        self.grid_forget()



class PredictBatchPredictionFrame(ctk.CTkFrame):
    """
    A frame that et user evaluate models performance based on existed dataset.
    """

    def __init__(self, container):
        # Initialize values
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        self.button_setting = {
            'corner_radius': 10, 
            'compound': 'left',  
            'anchor': 'w', 
            'border_spacing': 10, 
            'height': 25,
            'text_color': 'gray90', 
            'hover_color': 'gray30', 
            'fg_color': '#3d85c6',
            'font': ctk.CTkFont(size=18, weight="bold")
        }

        # Setting plotting appearance
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

        palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
        palette_rgb = [hex_to_rgb(x) for x in palette]
        cmap = mpl_colors.ListedColormap(palette_rgb)
        colors = cmap.colors
        bg_color= '#fdfcf6'

        custom_params = {
            "axes.spines.right": False,
            "axes.spines.top": False,
            'grid.alpha':0.3,
            # 'figure.figsize': (16, 6),
            'axes.titlesize': 'Large',
            'axes.labelsize': 'Large',
            'figure.facecolor': bg_color,
            'axes.facecolor': bg_color
        }

        sns.set_theme(
            style='whitegrid',
            palette=sns.color_palette(palette),
            rc=custom_params
        )

        # Add 2x2 grid
        self.grid_columnconfigure(0, weight=4)
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=2)

        # Add countplot
        self.figure_countplot, self.ax_countplot = plt.subplots()
        self.canvas_countplot = FigureCanvasTkAgg(self.figure_countplot, self)
        self.canvas_countplot.draw()
        self.canvas_countplot.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Add bottom frame
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky='nsew')
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        # Add Confusion matrix
        self.figure_confusion_mat, self.ax_confusion_mat = plt.subplots()
        self.canvas_confusion_mat = FigureCanvasTkAgg(self.figure_confusion_mat, self.bottom_frame)
        self.canvas_confusion_mat.draw()
        self.canvas_confusion_mat.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Add image frame
        # self.image_frame = ctk.CTkFrame(self.bottom_frame)
        # self.image_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Add Image Viewer
        self.image_viewer = ImageViewer(self.bottom_frame)
        self.image_viewer.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Add button frame and buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=0, column=1, sticky='nsew')
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        self.button_frame.grid_rowconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(3, weight=1)

        # Add result box
        self.result_box = ctk.CTkTextbox(self, font=ctk.CTkFont(size=15))
        self.result_box.insert(tk.INSERT, 'RESULT BOX\n')
        self.result_box.grid(row=1, column=1, sticky='nsew')
        self.result_box.configure(state="disabled")

        # Add dataset information
        self.dataset_name = None
        self.directory = None
        self.meta_data = None

        # Add dataset label
        self.data_image_icon = ctk.CTkImage(Image.open(self.image_path + 'data_icon.png'), size=(30, 30))
        self.dataset_label = ctk.CTkLabel(self.button_frame, font=ctk.CTkFont(size=20, weight="bold"), compound='left', image=self.data_image_icon, height=60)
        self.dataset_label.grid(row=0, column=0, padx=10, pady=10, sticky='new')

        # Add choose model
        self.model_menu = ctk.CTkComboBox(self.button_frame, height=30, values=config.MODEL_LIST, command=self.model_menu_event)
        self.model_menu.set('Choose Algorithm')
        self.model_menu.grid(row=1, column=0, padx=10, pady=15, sticky='ew')

        # Model attributes
        self.model_type = None
        self.model = None
        self.model_name = ''
        self.data = None

        # Add summary button
        self.summary_icon = ctk.CTkImage(Image.open(self.image_path + 'summary-icon.jpg'), size=(28, 28))
        self.summary_button = ctk.CTkButton(self.button_frame, image=self.summary_icon, text='Predict', 
                                         command=self.summary_button_event, **self.button_setting)
        self.summary_button.grid(row=2, column=0, padx=10, pady=15, sticky='ew')

        # Add label
        self.robot_icon = ctk.CTkImage(Image.open(self.image_path + 'gpt.png'), size=(200, 175))
        self.robot_label = ctk.CTkLabel(self.button_frame, image=self.robot_icon, text='', compound='bottom')
        self.robot_label.grid(row=3, column=0, sticky='nsew')



    def update_label(self):
        # Update the displayed label
        self.dataset_label.configure(text=f'    Dataset: {self.dataset_name}')

        # Plot the countplot
        self.meta_data = pd.read_csv(self.directory + 'meta.csv')
        sns.barplot(data=self.meta_data, x='Count', y='Label', ax=self.ax_countplot, orient='h', errorbar=None)
        self.ax_countplot.set_title('Dataset Labels Summary')
        self.canvas_countplot.draw()

        # Set image viewer to display the first image
        self.data = pd.read_csv(self.directory + 'data.csv')
        self.image_viewer.images = [self.data.iloc[i, :-1].values.reshape((28, 28)) for i in range(len(self.data))]
        self.image_viewer.images_label = self.data.iloc[:, -1].values
        self.image_viewer.show_image()


    def model_menu_event(self, choice):
        self.model_type = choice
        
        # Open model in the dir
        result = filedialog.askopenfilename(initialdir=config.MODEL_PATH + self.model_type)
        if len(result) == 0:
            showerror(message="Please choose a model", title='Error')
        else:
            # Load the model
            if self.model_type == 'CNN':
                self.model = load_model(result)
            else:
                self.model = joblib.load(result)


    def summary_button_event(self):
        if self.model is None:
            showerror(message='Please specify a model first.', title='Model Error')
            return

        # Add Confusion matrix
        self.figure_confusion_mat, self.ax_confusion_mat = plt.subplots()
        self.canvas_confusion_mat = FigureCanvasTkAgg(self.figure_confusion_mat, self.bottom_frame)
        self.canvas_confusion_mat.draw()
        self.canvas_confusion_mat.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Get the dataset
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values

        # Get predicted values
        if self.model_type == 'CNN':
            X = X.reshape((-1, 28, 28, 1))
            y_pred = self.model.predict(x=X).argmax(axis=1)
        else:
            y_pred = self.model.predict(X)

        # Plot the confusion matrix
        sns.heatmap(
            confusion_matrix(y, y_pred),
            annot=True,
            cmap='viridis',
            fmt='',
            ax=self.ax_confusion_mat
        )

        self.ax_confusion_mat.set_xticklabels(list(range(10)))
        self.ax_confusion_mat.set_yticklabels(list(range(10)))
        self.ax_confusion_mat.set_ylabel('True Label')
        self.ax_confusion_mat.set_xlabel('Prediction Label')
        self.ax_confusion_mat.set_title('Confusion Matrix')
        self.canvas_confusion_mat.draw()

        # Display accuracy score
        accuracy = np.round(accuracy_score(y, y_pred), 4)
        self.insert_text(f'Accuracy: {accuracy}')


    def insert_text(self, text):
        """
        Function to insert text into result box
        """
        self.result_box.configure(state="normal")
        self.result_box.insert(tk.END, '\n' + text)
        self.result_box.configure(state="disabled")


    def close(self):
        # Recreate Confusion matrix
        self.figure_confusion_mat, self.ax_confusion_mat = plt.subplots()
        self.canvas_confusion_mat = FigureCanvasTkAgg(self.figure_confusion_mat, self.bottom_frame)
        self.canvas_confusion_mat.draw()
        self.canvas_confusion_mat.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Recreate countplot
        self.figure_countplot, self.ax_countplot = plt.subplots()
        self.canvas_countplot = FigureCanvasTkAgg(self.figure_countplot, self)
        self.canvas_countplot.draw()
        self.canvas_countplot.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Delete the result box
        self.result_box.configure(state="normal")
        self.result_box.delete('2.0', 'end')
        self.result_box.configure(state="disabled")

        self.grid_forget()     