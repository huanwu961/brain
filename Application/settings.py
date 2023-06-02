import customtkinter as ctk


class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master, corner_radius=10):
        super().__init__(master, corner_radius=corner_radius)
        self.label = ctk.CTkLabel(self, text="Settings")
        self.label.pack()

