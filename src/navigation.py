import tkinter as tk
import customtkinter as ctk
from PIL import Image
import config

class NavigationFrame(ctk.CTkFrame):
    """
    Navigation Frame that let user to change between frames
    """

    def __init__(self, container):
        # Setting appearance
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        

        # Load all icons and images
        self.image_path = config.IMAGE_PATH
        self.logo_image = ctk.CTkImage(Image.open(self.image_path + 'CustomTkinter_logo_single.png'), size=(30, 30))
        self.image_icon_image = ctk.CTkImage(Image.open(self.image_path + 'image_icon_draw.png'), size=(24, 24))
        self.home_icon_image = ctk.CTkImage(Image.open(self.image_path + 'home_icon.png'), size=(24, 24))
        self.train_icon_image = ctk.CTkImage(Image.open(self.image_path + 'train_icon.png'), size=(24, 24))
        self.predict_icon_image = ctk.CTkImage(Image.open(self.image_path + 'predict_icon.png'), size=(24, 24))
        self.exit_icon_image = ctk.CTkImage(Image.open(self.image_path + 'exit_icon.png'), size=(24, 24))

        # Button appearance setting
        self.button_setting = {
            'corner_radius': 0,
            'height': 60,
            'border_spacing': 10,
            'fg_color': 'transparent',
            'text_color': 'gray90',
            'hover_color': 'gray30',
            'anchor': 'w',
            'font': ctk.CTkFont(size=17, weight="bold")
        }

        # Add label
        self.label = ctk.CTkLabel(self, text='   MENU', image=self.logo_image, height=90,
                                  compound="left", font=ctk.CTkFont(size=20, weight="bold"))
        self.label.grid(row=0, column=0, padx=20, pady=20)

        # Add Home button
        self.home_button = ctk.CTkButton(self, text='Home', image=self.home_icon_image,
                                         command=self.home_button_event, **self.button_setting)
        self.home_button.grid(row=1, column=0, sticky="ew")

        # Add create Image button
        self.draw_button = ctk.CTkButton(self, text='Draw', image=self.image_icon_image,
                                         command=self.draw_button_event, **self.button_setting)
        self.draw_button.grid(row=2, column=0, sticky="ew")

        # Add training button
        self.train_button = ctk.CTkButton(self, text='Train', image=self.train_icon_image,
                                          command=self.train_button_event, **self.button_setting)
        self.train_button.grid(row=3, column=0, sticky="ew")

        # Add Live Prediction button
        self.predict_button = ctk.CTkButton(self, text='Predict', image=self.predict_icon_image,
                                            command=self.predict_button_event, **self.button_setting)
        self.predict_button.grid(row=4, column=0, sticky="ew")

        # # Add setting button
        # self.setting_button = ctk.CTkButton(self, text='Setting', image=self.setting_icon_image,
        #                                     command=self.setting_button_event, **self.button_setting)
        # self.setting_button.grid(row=5, column=0, sticky="ew")

        # Add exit button
        self.exit_button = ctk.CTkButton(self, text='Exit', image=self.exit_icon_image,
                                         command=self.container.destroy, **self.button_setting)
        self.exit_button.grid(row=5, column=0, sticky="ew")



    def select_frame_by_name(self, name):
        """
        Changing between frames based on button pressed
        """

        # Change button appearance base on our current frame
        self.draw_button.configure(fg_color='gray25' if name == 'draw' else 'transparent')
        self.predict_button.configure(fg_color='gray25' if name == 'predict' else 'transparent')
        self.home_button.configure(fg_color='gray25' if name == 'home' else 'transparent')
        # self.setting_button.configure(fg_color='gray25' if name == 'setting' else 'transparent')
        self.train_button.configure(fg_color='gray25' if name == 'train' else 'transparent')

        # Close all frames
        self.container.home_frame.grid_forget()
        self.container.draw_start_frame.close()
        self.container.draw_main_frame.close()
        self.container.predict_frame.close()
        self.container.predict_live_frame.close()
        self.container.predict_batch_frame.close()
        # self.container.setting_frame.grid_forget()
        self.container.train_frame.close()

        # Show selected frame
        if name == 'home':
            self.container.home_frame.grid(row=0, column=1, sticky='nsew')
            
        if name == 'draw':
            self.container.draw_start_frame.grid(row=0, column=1, sticky='nsew')

        if name == 'predict':
            self.container.predict_frame.grid(row=0, column=1, sticky='nsew')

        # if name == 'setting':
        #     self.container.setting_frame.grid(row=0, column=1, sticky='nsew')

        if name == 'train':
            self.container.train_frame.grid(row=0, column=1, sticky='nsew')


    def home_button_event(self):
        self.select_frame_by_name('home')


    def draw_button_event(self):
        self.select_frame_by_name('draw')


    def predict_button_event(self):
        self.select_frame_by_name('predict')


    def train_button_event(self):
        self.select_frame_by_name('train')


    # def setting_button_event(self):
    #     self.select_frame_by_name('setting')