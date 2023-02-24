import tkinter as tk
import customtkinter as ctk

import config

from tkVideoPlayer import TkinterVideo


class HomeFrame(ctk.CTkFrame):
    """
    HomeFrame that displays the video of the robot
    """

    def __init__(self, container):
        # Setting appearance
        self.container = container
        ctk.set_appearance_mode(self.container.appearance_mode)
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        
        # Display the video
        self.home_video = TkinterVideo(self, scaled=True)
        self.home_video.load(self.image_path + 'welcome1.mp4')
        self.home_video.pack(expand=True, fill='both')
        self.home_video.play()
        self.home_video.bind('<<Ended>>', self.loop_video)


    def loop_video(self, event):
        self.home_video.play()