import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import torch.nn as nn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Constants
SHARED_RULES = [
    {"id": 1, "name": "Regla de datos compartidos",
     "description": "Esta regla establece que los datos compartidos entre ventanas deben ser coherentes y actualizados."},
    {"id": 2, "name": "Regla de gráficas 3D",
     "description": "Define los parámetros de las gráficas 3D, como el rango permitido para cada eje y colores uniformes."},
    # Remaining rules truncated for brevity
]


class ActorCriticA2C(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActorCriticA2C, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        policy_logits = self.policy(x)
        value = self.value(x)
        return torch.softmax(policy_logits, dim=-1), value


# noinspection PyMethodMayBeStatic
class MultiWindowApp:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("Consola de Control - Panel de Control")
        self.main_window.geometry("400x300")

        self.results_window = None
        self.graph_window = None
        self.shared_data = "Sin datos"

        self.actor_critic = ActorCriticA2C(10, 5, 128)  # Example initialization
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters())
        self.experiences = {"log_probs": [], "values": [], "rewards": []}

        # Buttons
        ttk.Button(self.main_window, text="Mostrar Resultados", command=self.show_results_window).pack(pady=10)
        ttk.Button(self.main_window, text="Mostrar Gráficas", command=self.show_graph_window).pack(pady=10)
        ttk.Button(self.main_window, text="Entrenar Agente", command=self.train_agent).pack(pady=10)

    @staticmethod
    def get_rules():
        """Returns the shared rules."""
        return SHARED_RULES

    def initialize_window(self, title, geometry):
        """Creates and returns a new window with the given title and size."""
        window = tk.Toplevel(self.main_window)
        window.title(title)
        window.geometry(geometry)
        return window

    def show_results_window(self):
        """Displays or updates a results window."""
        if self.results_window is None or not self.results_window.winfo_exists():
            self.results_window = self.initialize_window("Resultados", "400x300")
        # Update the results label
        tk.Label(self.results_window, text=self.shared_data, wraplength=300).pack(pady=20)

    def show_graph_window(self):
        """Displays a 3D graph window."""
        if self.graph_window is None or not self.graph_window.winfo_exists():
            self.graph_window = self.initialize_window("Gráficas 3D", "600x500")
            self.create_3d_graph(self.graph_window)

    def create_3d_graph(self, window):
        """Generates a 3D scatter plot and embeds it into a Tkinter window."""
        FIG_SIZE = (5, 4)  # Figure size in inches
        DPI = 100  # Dots per inch for figure resolution
        DATA_POINTS = 100  # Number of random data points
        COLOR_MAP = "viridis"  # Colormap for the scatter plot
    
        def generate_random_data(points):
            """Generates random (x, y, z) data points."""
            return np.random.random(points), np.random.random(points), np.random.random(points)
    
        # Step 1: Setup plot
        figure = Figure(figsize=FIG_SIZE, dpi=DPI)
        axes = figure.add_subplot(111, projection="3d")  # Create a 3D subplot
    
        x, y, z = generate_random_data(DATA_POINTS)
        axes.scatter(x, y, z, c=z, cmap=COLOR_MAP, marker="o")
        axes.set_title("3D Graph")
        axes.set_xlabel("X Axis")
        axes.set_ylabel("Y Axis")
        axes.set_zlabel("Z Axis")
    
        # Step 2: Integrate with Tkinter window
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def train_agent(self):
        """Trains the A2C agent."""
        for epoch in range(100):
            # Simulate state and action
            state = torch.tensor([[np.random.randint(0, 10)]], dtype=torch.float32)
            action_probs, value = self.actor_critic(state)
            action = torch.multinomial(action_probs, 1).item()
            reward = np.random.uniform(-1, 1)

            # Store experiences
            log_prob = torch.log(action_probs[0, action])
            self.experiences["log_probs"].append(log_prob)
            self.experiences["values"].append(value)
            self.experiences["rewards"].append(reward)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}: Acción {action}, Recompensa {reward:.2f}")

        self.shared_data = "Entrenamiento completado"
        messagebox.showinfo("Entrenamiento", self.shared_data)

    def run(self):
        """Runs the application."""
        try:
            self.main_window.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def process_command(self, command):
        """Processes a command related to the application workflow."""
        if command == "entrenar":
            self.train_agent()
            output = "Agente entrenado con éxito."
        elif command == "graficar":
            if self.graph_window is None or not self.graph_window.winfo_exists():
                self.show_graph_window()
            output = "Ventana de gráficas desplegada."
        elif command == "resultados":
            if self.results_window is None or not self.results_window.winfo_exists():
                self.show_results_window()
            output = "Ventana de resultados desplegada."
        elif command == "ayuda":
            output = "Comandos válidos: entrenar, graficar, resultados, ayuda."
        else:
            output = f"Comando desconocido: {command}"
        return output

if __name__ == "__main__":
    app = MultiWindowApp()
    app.run()
