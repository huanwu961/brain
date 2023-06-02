import customtkinter as ctk
import tkinter as tk

import home
import build
import run
import settings
import monitor


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.project_root = ""
        '''
         # Load the config file
        if not os.path.exists(os.path.expanduser('~/.config/brain/config.json')):
            self.create_project_root()

        with open(os.path.expanduser('~/.config/brain/config.json'), 'r+') as config_file:
            self.project_root = json.load(config_file)['project_root']
            self.auto_save_config = json.load(config_file)['auto_save']

        # find all the brains
        self.brains = []
        for file in os.listdir(self.project_root):
            if os.path.exists(os.path.join(self.project_root, file, 'config.json')):
                self.brains.append(file)
        '''
        # Create the main window
        self.title("Brain")
        self.geometry("640x480")

        # Configure the grid of the main window
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((1, 2), weight=1)

        # Create menu in the side bar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Brain", font=("Arial", 20))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        self.home_button = ctk.CTkButton(self.sidebar_frame, text="Home", command=self.go_home)
        self.home_button.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.build_button = ctk.CTkButton(self.sidebar_frame, text="Build", command=self.go_build)
        self.build_button.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.run_button = ctk.CTkButton(self.sidebar_frame, text="Run", command=self.go_run)
        self.run_button.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.monitor_button = ctk.CTkButton(self.sidebar_frame, text="Monitor", command=self.go_monitor)
        self.monitor_button.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.settings_button = ctk.CTkButton(self.sidebar_frame, text="Settings", command=self.go_settings)
        self.settings_button.grid(row=5, column=0, padx=20, pady=20, sticky="s")

        # Create the main frame
        self.home = home.HomeFrame(self)
        self.home.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.build = build.BuildFrame(self)
        self.build.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.run = run.RunFrame(self)
        self.run.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.monitor = monitor.MonitorFrame(self)
        self.monitor.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.settings = settings.SettingsFrame(self)
        self.settings.grid(row=0, column=1, rowspan=4, sticky="nsew")

    def go_home(self):
        self.home.tkraise()

    def go_build(self):
        self.build.tkraise()

    def go_run(self):
        self.run.tkraise()

    def go_settings(self):
        self.settings.tkraise()

    def go_monitor(self):
        self.monitor.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()
