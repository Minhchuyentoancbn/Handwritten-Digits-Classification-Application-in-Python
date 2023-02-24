import customtkinter as ctk
import os

import config

from pathlib import Path
from navigation import NavigationFrame
from home import HomeFrame
from draw import DrawMainFrame, DrawStartFrame
from predict import PredictStartFrame, PredictLivePredictionFrame, PredictBatchPredictionFrame
from train import TrainFrame



class App(ctk.CTk):
    """
    Our main Application
    """
    def __init__(self):
        # Setting color and appearance
        self.appearance_mode = 'dark'
        ctk.set_appearance_mode(self.appearance_mode)  # Modes: system (default), light, dark
        ctk.set_default_color_theme("dark-blue")

        # Initializing
        super().__init__()
        self.image_path = config.IMAGE_PATH

        # Setting title bar
        self.title('Handwritten digit regconition')
        # self.attributes('-topmost', 1)
        self.iconbitmap(self.image_path + 'icon.ico')

        # set grid layout 1x2
        self.geometry("1000x700")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Navigation frame
        self.navigation_frame = NavigationFrame(self)
        self.navigation_frame.grid(row=0, column=0, sticky='nsew')

        # Home Frame
        self.home_frame = HomeFrame(self)
        self.home_frame.grid(row=0, column=1, sticky='nsew')

        # Draw Frame
        self.draw_start_frame = DrawStartFrame(self)
        self.draw_main_frame = DrawMainFrame(self)

        # Predict Frame
        self.predict_frame = PredictStartFrame(self)
        self.predict_live_frame = PredictLivePredictionFrame(self)
        self.predict_batch_frame = PredictBatchPredictionFrame(self)

        # Train Frame
        self.train_frame = TrainFrame(self)

        # # Setting Frame
        # self.setting_frame = ctk.CTkFrame(self)



if __name__ == '__main__':
    # Create folders for models
    for model_type in config.MODEL_LIST:
        model_path = Path() / 'models' / model_type
        os.makedirs(model_path, exist_ok=True)

    # Run app
    app = App()
    app.mainloop()