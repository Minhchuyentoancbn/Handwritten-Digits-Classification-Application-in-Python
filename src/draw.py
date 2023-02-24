import csv
import os
import tkinter as tk
import customtkinter as ctk
import pandas as pd

import config

from PIL import Image, ImageDraw
from tkinter.messagebox import showerror
from tkinter import filedialog
from utils import process_image


class DrawStartFrame(ctk.CTkFrame):
    """
    Starting Frame when we choose draw function, use to let user be able to choose function
    """

    def __init__(self, container):
        # Setting appearance
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        self.button_setting = {
            'corner_radius': 10,
            'height': 110,
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

        # Add open button
        self.open_icon = ctk.CTkImage(Image.open(self.image_path + 'folder_icon.png'), size=(40, 40))
        self.open_button = ctk.CTkButton(self, image=self.open_icon, text='Open Existing\nDataset',
                                         command=self.open_button_event, **self.button_setting)
        self.open_button.grid(row=1, column=1)

         # Add create button
        self.create_icon = ctk.CTkImage(Image.open(self.image_path + 'create_icon.png'), size=(40, 40))
        self.create_button = ctk.CTkButton(self, image=self.create_icon, text='Create New\nDataset',
                                           command=self.create_button_event, **self.button_setting)
        self.create_button.grid(row=1, column=3)


    def open_button_event(self):
        """
        Open an existing dataset
        """
        # Ask for file
        result = filedialog.askdirectory(initialdir=config.USER_PATH)

        if len(result) > 0:
            # get the dataset name and its directory
            self.container.draw_main_frame.dataset_name = result.split(r'/')[-1]
            self.container.draw_main_frame.directory = result + r'//'
            
            # Close start frame and activate main frame
            self.grid_forget()  
            self.container.draw_main_frame.update_label()  # update dataset label
            self.container.draw_main_frame.grid(row=0, column=1, sticky='nsew')


    def create_button_event(self):
        """
        Create new directory for a new dataset
        """

        # Ask dataset name
        self.dialog = ctk.CTkInputDialog(text='Type in dataset\'s name', title='Input Name')
        result = self.dialog.get_input()

        if len(result) > 0:
            self.container.draw_main_frame.dataset_name = result
            self.container.draw_main_frame.directory = config.USER_PATH + result + r'//'

            try:
                os.mkdir(self.container.draw_main_frame.directory)
            except FileExistsError:
                showerror(message='Dataset already exists. Please try another name', title='Name Error')
            else:
                # Create a file that contains summaries about the dataset
                meta_data = pd.DataFrame({'Label': range(10), 'Count': 0})
                meta_data.to_csv(self.container.draw_main_frame.directory + 'meta.csv', index=False)
                headers = [f'pixel {i}' for i in range(784)] + ['label']
                summary_data = pd.DataFrame(columns=headers)
                summary_data.to_csv(self.container.draw_main_frame.directory + 'data.csv', index=False)

                # Close start frame and activate main frame
                self.grid_forget()
                self.container.draw_main_frame.update_label()
                self.container.draw_main_frame.grid(row=0, column=1, sticky='nsew')
        else:
            showerror(message="Length must be greater than zero", title='Name Error')


    def close(self):
        # close the frame
        self.grid_forget()
        

class DrawMainFrame(ctk.CTkFrame):
    """
    The Main Frame that let user draw, label and save images.
    """

    def __init__(self, container):
        # Initialize the value
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        self.directory = None
        self.dataset_name = None

        # Add 2 x 2 grid
        self.grid_columnconfigure(0, weight=4)
        # self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=2)

        # Add canvas
        self.canvas = ctk.CTkCanvas(self, bg='#bcbcbc', height=config.CANVAS_HEIGHT, width=config.CANVAS_WIDTH)
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.canvas_width = self.canvas.winfo_screenwidth()
        self.canvas_height = self.canvas.winfo_screenheight()

        # Add clear button
        self.clear_button = ctk.CTkButton(self.canvas, text='clear',  width=30, corner_radius=0, command=self.clear_canvas)
        self.clear_button.place(x=0, y=0)

        # Add button frame and buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=0, column=1, sticky='nsew')
        self.button_frame.grid_rowconfigure(5, weight=1)
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        self.button_frame.grid_rowconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(3, weight=1)
        self.button_frame.grid_rowconfigure(4, weight=1)

        # Add dataset label
        self.data_image_icon = ctk.CTkImage(Image.open(self.image_path + 'data_icon.png'), size=(30, 30))
        self.dataset_label = ctk.CTkLabel(self.button_frame, font=ctk.CTkFont(size=20, weight="bold"), compound='left', image=self.data_image_icon, height=60)
        self.dataset_label.grid(row=0, column=0, padx=10, pady=10, sticky='new')

        # Add draw button
        self.paint_icon = ctk.CTkImage(Image.open(self.image_path + 'paint_icon.png'), size=(25, 25))
        self.draw_button = ctk.CTkButton(self.button_frame, corner_radius=10, image=self.paint_icon,
                                         compound='left', text='Draw', anchor='w', border_spacing=10, height=25,
                                         text_color='gray90', hover_color='gray30', fg_color='#3d85c6',
                                         font=ctk.CTkFont(size=18, weight="bold"), command=self.draw_button_event)
        self.draw_button.grid(row=1, column=0, padx=10, pady=15, sticky='ew')

        # Adjust width
        self.pen_width = 20
        self.thick_combobox = ctk.CTkComboBox(self.button_frame, values=['Pen Thickness: 10', 'Pen Thickness: 20', 'Pen Thickness: 30', 'Pen Thickness: 40'], 
                                              command=self.update_pen_width, height=30)
        self.thick_combobox.set('Choose Pen Thickness')
        self.thick_combobox.grid(row=3, column=0, padx=10, pady=15, sticky='ew')

        # Add Label Entry
        self.label_entry = ctk.CTkEntry(self.button_frame, corner_radius=10, placeholder_text='Enter the image label', height=30)
        self.label_entry.grid(row=4, column=0, padx=10, pady=15, sticky='ew')

        # Image backend
        self.current_image = Image.new('L', (self.canvas_width, self.canvas_height))
        self.draw = ImageDraw.Draw(self.current_image)

        # Add Save Button
        self.save_icon = ctk.CTkImage(Image.open(self.image_path + 'save_icon.png'), size=(25, 25))
        self.save_button = ctk.CTkButton(self.button_frame, corner_radius=10, image=self.save_icon,
                                         compound='left', text='Save', anchor='w', border_spacing=10, height=25,
                                         text_color='gray90', hover_color='gray30', fg_color='#3d85c6',
                                         font=ctk.CTkFont(size=18, weight="bold"), command=self.save_button_event)
        self.save_button.grid(row=2, column=0, padx=10, pady=15, sticky='ew')

        # Add label
        self.artist_frame = ctk.CTkFrame(self.button_frame)
        self.artist_frame.grid(row=5, column=0, sticky='nsew')
        self.artist_icon = ctk.CTkImage(Image.open(self.image_path + 'artist_icon.png'), size=(100, 100))
        self.artist_label = ctk.CTkLabel(self.artist_frame, image=self.artist_icon, text='', compound='bottom')
        # self.artist_label.grid(row=5, column=0, padx=10, sticky='nsew')
        self.artist_label.pack(fill='both', expand=True)

        # Add message box
        self.message_box = ctk.CTkTextbox(self, font=ctk.CTkFont(size=15))
        self.message_box.insert(tk.INSERT, 'MESSAGE BOX\n')
        self.message_box.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.message_box.configure(state="disabled")
        


    def update_label(self):
        """
        Update the dataset label
        """
        self.dataset_label.configure(text=f'    Dataset: {self.dataset_name}')


    def clear_canvas(self):
        """
        Clear canvas and underlying image
        """
        self.canvas.delete('all')
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
        # print(f'x = {x}, y = {y}')
        self.canvas.create_line((self.lastx, self.lasty, x, y), width=self.pen_width, fill='black')
        self.draw.line([self.lastx, self.lasty, x, y], 255, width=self.pen_width)
        self.lastx, self.lasty = x, y


    def update_pen_width(self, choice):
        self.pen_width = int(choice.split()[-1])


    def save_button_event(self):
        # Get the image label
        label = self.label_entry.get()
        if len(label) == 0:
            showerror(message='Please enter the label.', title='Label Error')
        else:
            try:
                digit = int(label)
            except:
                showerror(message='Label can only be a number from 0 to 9.', title='Label Error')
            else:
                if digit > 9 or digit < 0:
                    showerror(message='Label can only be a number from 0 to 9.', title='Label Error')
                else:
                    # Get image ID and save
                    meta = pd.read_csv(self.directory + 'meta.csv')
                    num_images = meta['Count'].sum()
                    image_id = num_images + 1
                    image_path = self.directory + f'{image_id}-{digit}.png'
                    self.current_image.save(image_path)

                    # Initialize new backend image
                    self.current_image = Image.new('L', (self.canvas_width, self.canvas_height))
                    self.draw = ImageDraw.Draw(self.current_image)

                    # Update metadata
                    meta.loc[meta['Label'] == digit, 'Count'] += 1
                    meta.to_csv(self.directory + 'meta.csv', index=False)

                    # Update the dataset
                    img_array = process_image(image_path)
                    data = img_array.reshape((-1)).tolist()
                    data.append(digit)

                    with open(self.directory + 'data.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

                    # Update the message
                    self.message_box.configure(state="normal")
                    self.message_box.insert(tk.END, f'\n- Your image was saved successfully with ID = {image_id}. Label = {digit}')
                    self.message_box.configure(state="disabled")


    def close(self):
        # Clear the entry, message box and canvas
        self.label_entry.delete(0, 'end')
        self.label_entry.configure(placeholder_text='Enter the image label')

        self.message_box.configure(state="normal")
        self.message_box.delete('2.0', 'end')
        self.message_box.configure(state="disabled")

        self.clear_canvas()

        self.grid_forget()     