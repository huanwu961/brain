import customtkinter as ctk
import numpy as np
from PIL import Image

class MonitorFrame(ctk.CTkFrame):
    def __init__(self, master, corner_radius=10):
        super().__init__(master, corner_radius=corner_radius)
        self.label = ctk.CTkLabel(self, text="Monitor")
        self.label.pack()
        self.object_options = ['aaa', 'bbb']
        self.object_option_menu = ctk.CTkOptionMenu(self, values=self.object_options)
        self.object_option_menu.bind('<Button-1>', self.change_member)
        self.object_option_menu.pack()

        self.array_options = ['aaaa', 'bbbb']
        self.array_option_menu = ctk.CTkOptionMenu(self, values=self.array_options)
        self.array_option_menu.pack()

        self.view_button = ctk.CTkButton(self, text="view", command=self.view)
        self.view_button.pack()

        self.size_entry = ctk.CTkEntry(self)
        self.size_entry.pack()

        self.image_size = (640, 480)
        self.image = ctk.CTkImage(Image.fromarray(np.ones([100, 100])*255), size=self.image_size)
        self.image.pack()
        self.master = master

    def change_member(self, event):
        if self.object_option_menu.get() == 'aaa':
            self.array_options = ['a', 'b']
        else:
            self.array_options = ['aaa', 'bbb']
    def view(self, img):
        pass
