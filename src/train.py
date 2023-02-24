import tkinter as tk
import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing as mp
import time
import joblib

import config

from copy import deepcopy
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.messagebox import showerror
from tkinter import filedialog
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


class TrainFrame(ctk.CTkFrame):
    def __init__(self, container):
        # Setting appearance
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
            'height': 20,
            'text_color': 'gray90', 
            'hover_color': 'gray30', 
            'fg_color': '#3d85c6',
            'font': ctk.CTkFont(size=18, weight="bold")
        }
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

        # Add param frame
        self.param_frame = ctk.CTkFrame(self)
        self.param_frame.grid(row=0, column=0, sticky='nsew', padx=10)
        self.param_frame.grid_columnconfigure(0, weight=1)
        self.param_frame.grid_columnconfigure(1, weight=1)
        self.param_frame.grid_columnconfigure(2, weight=1)
        self.param_frame.grid_rowconfigure(0, weight=1)
        self.param_frame.grid_rowconfigure(1, weight=1)

        # add knn frame
        self.knn_frame = ctk.CTkFrame(self.param_frame)
        self.knn_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.knn_label = ctk.CTkLabel(master=self.knn_frame, text="k-Nearest Neighbors", font=ctk.CTkFont(size=20, weight="bold"))
        self.knn_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.k_choice_entry = ctk.CTkEntry(self.knn_frame, corner_radius=10, placeholder_text='Enter k', height=30)
        self.k_choice_entry.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        self.p_choice_entry = ctk.CTkEntry(self.knn_frame, corner_radius=10, placeholder_text='Enter p', height=30)
        self.p_choice_entry.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')
        self.metrics_menu = ctk.CTkComboBox(self.knn_frame, height=30, values=['uniform', 'distance'], command=self.knn_metrics_event)
        self.metrics_menu.set('Choose weights')
        self.metrics_menu.grid(row=4, column=0, padx=10, pady=10, sticky='nsew')

        # k-NN Params
        self.knn_k = 5
        self.knn_p = 2
        self.knn_metrics = 'uniform'


        # add svm frame
        self.svm_frame = ctk.CTkFrame(self.param_frame)
        self.svm_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        self.svm_label = ctk.CTkLabel(master=self.svm_frame, text="Support Vector Machine", font=ctk.CTkFont(size=20, weight="bold"))
        self.svm_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.svm_c_choice_entry = ctk.CTkEntry(self.svm_frame, corner_radius=10, placeholder_text='Enter C', height=30)
        self.svm_c_choice_entry.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        self.svm_penalty_menu = ctk.CTkComboBox(self.svm_frame, height=30, values=['l1', 'l2'], command=self.svm_penalty_event)
        self.svm_penalty_menu.set('Choose penalty')
        self.svm_penalty_menu.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')
        self.loss_menu = ctk.CTkComboBox(self.svm_frame, height=30, values=['squared_hinge', 'hinge'], command=self.svm_loss_event)
        self.loss_menu.set('Choose loss')
        self.loss_menu.grid(row=4, column=0, padx=10, pady=10, sticky='nsew')    

        # SVM parmas  
        self.svm_C = 1.0
        self.svm_loss = 'squared_hinge'
        self.svm_penalty = 'l2'

        # add lr frame
        self.lr_frame = ctk.CTkFrame(self.param_frame)
        self.lr_frame.grid(row=0, column=2, sticky='nsew', padx=10, pady=10)
        self.lr_label = ctk.CTkLabel(master=self.lr_frame, text="Logistics Regression", font=ctk.CTkFont(size=20, weight="bold"))
        self.lr_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.lr_c_choice_entry = ctk.CTkEntry(self.lr_frame, corner_radius=10, placeholder_text='Enter C', height=30)
        self.lr_c_choice_entry.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
        self.lr_penalty_menu = ctk.CTkComboBox(self.lr_frame, height=30, values=['l1', 'l2'], command=self.lr_penalty_event)
        self.lr_penalty_menu.set('Choose penalty')
        self.lr_penalty_menu.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

        # Logistics Regression params
        self.lr_C = 1.0
        self.lr_penalty = 'l2'

        # Add button frame and buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=0, column=1, sticky='nsew')
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        self.button_frame.grid_rowconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(3, weight=1)
        self.button_frame.grid_rowconfigure(4, weight=1)

        # Add train button
        self.model_image_icon = ctk.CTkImage(Image.open(self.image_path + 'ai_icon.png'), size=(25, 25))
        self.train_button = ctk.CTkButton(self.button_frame, image=self.model_image_icon, text='Train', 
                                            **self.button_setting, command=self.train_button_event)
        self.train_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Add learning curve
        self.paint_icon = ctk.CTkImage(Image.open(self.image_path + 'paint_icon.png'), size=(25, 25))
        self.draw_button = ctk.CTkButton(self.button_frame, image=self.paint_icon,
                                         text='Learning curve', command=self.draw_button_event, 
                                         **self.button_setting)
        self.draw_button.grid(row=1, column=0, padx=10, pady=10, sticky='ew')


        # Add save button
        self.save_icon = ctk.CTkImage(Image.open(self.image_path + 'save_icon.png'), size=(25, 25))
        self.save_button = ctk.CTkButton(self.button_frame, image=self.save_icon,
                                         text='Save', command=self.save_button_event, **self.button_setting)
        self.save_button.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        # Add choose model
        self.model_menu = ctk.CTkComboBox(self.button_frame, height=30, values=[name for name in config.MODEL_LIST if name != 'CNN'], 
                                          command=self.model_menu_event)
        self.model_menu.set('Choose Current Model')
        self.model_menu.grid(row=3, column=0, padx=10, pady=10, sticky='ew')
        
        # Add model name
        self.name_entry = ctk.CTkEntry(self.button_frame, corner_radius=10, placeholder_text='Enter the model name', height=30)
        self.name_entry.grid(row=4, column=0, padx=10, pady=10, sticky='ew')

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

        # Add learning curve
        self.figure_learning_curve, self.ax_learning_curve = plt.subplots()
        self.canvas_learning_curve = FigureCanvasTkAgg(self.figure_learning_curve, self.bottom_frame)
        self.canvas_learning_curve.draw()
        self.canvas_learning_curve.get_tk_widget().grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Add result box
        self.result_box = ctk.CTkTextbox(self, font=ctk.CTkFont(size=15))
        self.result_box.insert(tk.INSERT, 'RESULT BOX\n')
        self.result_box.grid(row=1, column=1, sticky='nsew')
        self.result_box.configure(state="disabled")

        # Add model
        self.model_type = None
        self.model = None

        # Add progress bar
        self.progress_bar = ctk.CTkProgressBar(self.param_frame, mode='indeterminnate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        self.queue = mp.Queue()


    def close(self):
        # Progress bar
        self.progress_bar.set(0)
        self.progress_bar.stop()

        # Result box
        self.result_box.configure(state="normal")
        self.result_box.delete('2.0', 'end')
        self.result_box.configure(state="disabled")

       # Recreate Confusion matrix
        self.figure_confusion_mat, self.ax_confusion_mat = plt.subplots()
        self.canvas_confusion_mat = FigureCanvasTkAgg(self.figure_confusion_mat, self.bottom_frame)
        self.canvas_confusion_mat.draw()
        self.canvas_confusion_mat.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Recreate learning curve
        self.figure_learning_curve, self.ax_learning_curve = plt.subplots()
        self.canvas_learning_curve = FigureCanvasTkAgg(self.figure_learning_curve, self.bottom_frame)
        self.canvas_learning_curve.draw()
        self.canvas_learning_curve.get_tk_widget().grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        try:
            self.process.terminate()
        except:
            pass
        
        self.grid_forget()


    def train_button_event(self):
        if self.model_type is None:
            showerror(message="Please choose a model", title='Error')
            return

        # Recreate Confusion matrix
        self.figure_confusion_mat, self.ax_confusion_mat = plt.subplots()
        self.canvas_confusion_mat = FigureCanvasTkAgg(self.figure_confusion_mat, self.bottom_frame)
        self.canvas_confusion_mat.draw()
        self.canvas_confusion_mat.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Set up model and its hyperparameters
        if self.model_type == 'k-NN':
            k_entry = self.k_choice_entry.get()
            p_entry = self.p_choice_entry.get()

            if len(k_entry) == 0:
                self.knn_k = 5
            else:
                try:
                    k_entry = int(k_entry)
                except:
                    showerror(message='Please enter an integer for k.', title='Error')
                    return
                else:
                    self.knn_k = k_entry

            if len(p_entry) == 0:
                self.knn_p = 2
            else:
                try:
                    p_entry = int(p_entry)
                except:
                    showerror(message='Please enter an integer for p.', title='Error')
                    return
                else:
                    self.knn_p = p_entry

            print(self.knn_k)
            self.model = KNeighborsClassifier(n_neighbors=self.knn_k, p=self.knn_p, weights=self.knn_metrics)

        elif self.model_type == 'SVM':
            c_entry = self.svm_c_choice_entry.get()

            if len(c_entry) == 0:
                self.svm_C = 1.0
            else:
                try:
                    c_entry = float(c_entry)
                except:
                    showerror(message='Please enter a real number for C.', title='Error')
                    return
                else:
                    self.svm_C = c_entry

            self.model = LinearSVC(penalty=self.svm_penalty, loss=self.svm_loss, C=self.svm_C, random_state=42)

        else:
            c_entry = self.lr_c_choice_entry.get()

            if len(c_entry) == 0:
                self.lr_C = 1.0
            else:
                try:
                    c_entry = float(c_entry)
                except:
                    showerror(message='Please enter a real number for C.', title='Error')
                    return
                else:
                    self.lr_C = c_entry

            self.model = LogisticRegression(penalty=self.lr_penalty, C=self.lr_C, solver='saga', random_state=42, max_iter=2000)

        # Start training process
        self.queue = mp.Queue()
        self.process = mp.Process(target=train_model, args=(self.queue, deepcopy(self.model)))
        self.process.start()

        # Start progress bar
        self.progress_bar.start()
        self.after(1000, self.check_queue)
            
        
    def check_queue(self):
        if not self.queue.empty():
            # Get the results
            self.model, y_pred = self.queue.get()

            # Stop the progress bar
            self.progress_bar.stop()

            # Plot the confusion matrix
            test_df = pd.read_csv(config.MNIST_TEST)
            y_test = test_df['label'].copy()
            sns.heatmap(
                confusion_matrix(y_test, y_pred),
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

            # Display accuracy
            accuracy = np.round(accuracy_score(y_test, y_pred), 4)
            self.insert_text(f'Accuracy: {accuracy}')
            return

        self.after(1000, self.check_queue)


    def draw_button_event(self):
        if self.model_type is None:
            showerror(message="Please choose a model first.", title='Error')
            return

        if self.model is None:
            showerror(message="Please train a model first.", title='Error')
            return

        # Recreate learning curve
        self.figure_learning_curve, self.ax_learning_curve = plt.subplots()
        self.canvas_learning_curve = FigureCanvasTkAgg(self.figure_learning_curve, self.bottom_frame)
        self.canvas_learning_curve.draw()
        self.canvas_learning_curve.get_tk_widget().grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Start running learning curve
        self.queue = mp.Queue()
        self.process = mp.Process(target=learning_curve_plot, args=(self.queue, deepcopy(self.model)))
        self.process.start()

        # Start process bar
        self.progress_bar.start()
        self.after(1000, self.check_queue2)


    def check_queue2(self):
        if not self.queue.empty():
            # Get the results
            train_size_abs, train_scores, test_scores = self.queue.get()

            # Stop the progress bar
            self.progress_bar.stop()

            # Plot the learning curve
            self.ax_learning_curve.plot(train_size_abs, train_scores.mean(axis=1), label='Training')
            self.ax_learning_curve.plot(train_size_abs, test_scores.mean(axis=1), label='Testing')
            self.ax_learning_curve.set_ylim(0, 1)
            self.ax_learning_curve.set_xlabel('Prediction Label')
            self.ax_learning_curve.set_ylabel('Accuracy')
            self.ax_learning_curve.set_title('Learning Curve')
            self.ax_learning_curve.legend()
            self.canvas_learning_curve.draw()
            return


        self.after(1000, self.check_queue2)



    def save_button_event(self):
        # Get the model name
        model_name = self.name_entry.get()

        if len(model_name) == 0:
            showerror(message='Please enter model name first.', title='Error')
            return

        if self.model_type is None:
            showerror(message="Please choose a model first.", title='Error')
            return

        if self.model is None:
            showerror(message="Please train a model first.", title='Error')
            return

        # Save the model
        joblib.dump(self.model, config.MODEL_PATH + self.model_type + '\\' + model_name)
        self.insert_text('Save successfully')
        

    def model_menu_event(self, choice):
        self.model_type = choice
    

    def knn_metrics_event(self, choice):
        self.knn_metrics = choice


    def svm_loss_event(self, choice):
        self.svm_loss = choice


    def svm_penalty_event(self, choice):
        self.svm_penalty = choice


    def lr_penalty_event(self, choice):
        self.lr_penalty = choice


    def insert_text(self, text):
        """
        Function to insert text to result box
        """
        self.result_box.configure(state="normal")
        self.result_box.insert(tk.END, '\n' + text)
        self.result_box.configure(state="disabled")      



def train_model(queue, model):
    # Get the dataset
    train_df = pd.read_csv(config.MNIST_TRAIN)
    y_train = train_df['label'].copy()
    X_train = train_df.drop(columns='label').values / 255
    test_df = pd.read_csv(config.MNIST_TEST)
    y_test = test_df['label'].copy()
    X_test = test_df.drop(columns='label').values / 255

    print('Start Training')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Stop training')

    # Return values
    queue.put((model, y_pred))


def learning_curve_plot(queue, model):
    # Get the dataset
    train_df = pd.read_csv(config.MNIST_TRAIN)
    y_train = train_df['label'].copy()
    X_train = train_df.drop(columns='label').values / 255

    print('Start Plotting')
    train_size_abs, train_scores, test_scores = learning_curve(
        model, X_train, y_train, scoring='accuracy', random_state=42, cv=2
    )
    print('Stop Plotting')

    # Return values
    queue.put((train_size_abs, train_scores, test_scores))