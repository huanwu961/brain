import customtkinter as ctk


class MonitorFrame(ctk.CTkFrame):
    def __init__(self, master, corner_radius=10):
        super().__init__(master, corner_radius=corner_radius)
        self.label = ctk.CTkLabel(self, text="Monitor")
        self.label.pack()
