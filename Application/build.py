import customtkinter as ctk


class BuildFrame(ctk.CTkFrame):
    def __init__(self, master, corner_radius=10):
        super().__init__(master, corner_radius=corner_radius)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.location_entry = ctk.CTkEntry(self, width=300)
        self.location_entry.grid(row=0, padx=20, pady=(20, 0), sticky="ns")

        self.config_text_box = ctk.CTkTextbox(self)
        self.config_text_box.grid(padx=20, pady=20, sticky="nesw")

        self.config_load_button = ctk.CTkButton(self, text="Load", command=self.load)
        self.config_load_button.grid(row=2, padx=20, pady=(0, 20), sticky="sw")

        self._build_button = ctk.CTkButton(self, text="Build", command=self.build)
        self._build_button.grid(row=2, padx=20, pady=(0, 20), sticky="es")

    def build(self):
        pass

    def load(self):
        pass
