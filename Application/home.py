import customtkinter as ctk


class HomeFrame(ctk.CTkFrame):
    def __init__(self, master, corner_radius=10):
        super().__init__(master, corner_radius=corner_radius)
        self.home_label = ctk.CTkLabel(self, text="Home")
        self.home_label.pack()