import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from environment import nn


class MultiWindowApp:
    # Constants for window dimensions and titles
    WINDOW_GEOMETRY_MAIN = "400x300"
    WINDOW_GEOMETRY_GRAPH = "600x500"
    WINDOW_RESULTS_TITLE = "Ventana de Resultados"
    WINDOW_GRAPH_TITLE = "Ventana de Gráfica 3D"
    WINDOW_COMMAND_TITLE = "Ventana de Comandos"

    def __init__(self):
        # Ventana principal
        self.main_window = tk.Tk()
        self.main_window.title("Panel de Control - Figuras 3D")
        self.main_window.geometry(self.WINDOW_GEOMETRY_MAIN)
        self.command_entry = None
        self.results_window = self.shared_data = None

        # Shared data and windows
        self.shared_data = "Sin datos enviados aún"
        self.results_window = None
        self.result_label = None
        self.graph_window = None

        # UI Components
        ttk.Button(self.main_window, text="Abrir Ventana de Resultados", command=self.open_results_window).pack(pady=10)
        ttk.Button(self.main_window, text="Abrir Ventana de Gráficas", command=self.open_graph_window).pack(pady=10)

        # Corrected: Use a callable reference without parentheses
        ttk.Button(self.main_window, text="Entrenamiento", command=nn).pack(pady=10)

        # Placeholder entry field and bind corrected
        self.command_entry = ttk.Entry(self.main_window, width=30)
        self.command_entry.pack(pady=10)
        self.command_entry.insert(0, "")
        # Corrected: Binds require a callable function reference
        self.command_entry.bind("<Return>", lambda event: nn)  # Example for <Return> key event

    @staticmethod
    def _is_window_open(window):
        """Checks if a window is already open."""
        return window is not None and window.winfo_exists()

    @staticmethod
    def _generate_3d_data():
        """Generates random 3D data for visualization."""
        return np.random.rand(100), np.random.rand(100), np.random.rand(100)

    @staticmethod
    def _create_3d_scatter_plot(parent, x_data, y_data, z_data):
        """Creates and displays a 3D scatter plot in the given parent window."""
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_data, y_data, z_data, c=z_data, cmap="viridis", marker="o")
        ax.set_title("Gráfica 3D")
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_zlabel("Eje Z")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_window(self, title, geometry):
        """Creates a new window with the specified title and geometry."""
        new_window = tk.Toplevel(self.main_window)
        new_window.title(title)
        new_window.geometry(geometry)
        return new_window

    def open_graph_window(self):
        """Opens the graph window with a 3D scatter plot."""
        if not self._is_window_open(self.graph_window):
            self.graph_window = self._create_window(self.WINDOW_GRAPH_TITLE, self.WINDOW_GEOMETRY_GRAPH)
            x_data, y_data, z_data = self._generate_3d_data()
            self._create_3d_scatter_plot(self.graph_window, x_data, y_data, z_data)

    def open_results_window(self):
        """Opens the results window for shared data display."""
        if not self._is_window_open(self.results_window):
            self.results_window = self._create_window(self.WINDOW_RESULTS_TITLE, self.WINDOW_GEOMETRY_MAIN)

            # Lazy initialization of result_label
            self.result_label = ttk.Label(self.results_window, text=self.shared_data, wraplength=300)
            self.result_label.pack(pady=20)

    def send_data_to_results(self):
        """Sends the data from the entry box to the results window."""
        if not self._is_window_open(self.results_window):
            messagebox.showerror("Error", "Primero abre la ventana de resultados.")
            return

        data = self.command_entry.get().strip()
        if not data:
            data = "No se escribió nada."
        self.shared_data = f"Datos enviados: {data}"
        self.result_label.config(text=self.shared_data)

    def run(self):
        """Runs the application's main loop."""
        self.main_window.mainloop()


# Ejecutar la aplicación
if __name__ == "__main__":
    app = MultiWindowApp()
    app.run()
