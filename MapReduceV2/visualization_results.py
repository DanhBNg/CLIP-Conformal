"""
Interactive Visualization Selector
===================================
Allows users to select which method and visualization file to view
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_visualization_file(json_file):
    """Load visualization JSON file"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def visualize_samples(samples_data, fig=None):
    """Visualize samples with prediction sets"""
    if fig is None:
        n_samples = len(samples_data)
        fig = plt.figure(figsize=(16, 4.5 * n_samples))
    else:
        n_samples = len(samples_data)
    
    fig.clear()
    
    height_ratios = [1] * n_samples
    width_ratios = [2, 5]
    
    grid = plt.GridSpec(
        n_samples, 2,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.8,
        wspace=0.25,
        left=0.08,
        right=0.95,
        top=0.98,
        bottom=0.02
    )
    
    for idx, sample in enumerate(samples_data):
        image_path = Path(sample['image_path'])
        ground_truth = sample['class_name']
        pred_set = sample['prediction_set']
        confidence = sample['confidence_score']
        method = sample['method']
        alpha = sample['alpha']
        set_size = sample['set_size']
        
        # Display image
        ax_img = plt.subplot(grid[idx, 0])
        try:
            img = Image.open(image_path)
            ax_img.imshow(img)
        except Exception as e:
            ax_img.text(0.5, 0.5, f"Error loading image", ha='center', va='center', fontsize=10)
        
        ax_img.axis('off')
        
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        ax_img.set_title(f"Image {idx+1}", fontsize=13, fontweight='bold', pad=10)
        
        # Display prediction info
        ax_text = plt.subplot(grid[idx, 1])
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        
        # Mark ground truth in prediction set
        marked_predictions = []
        for pred in pred_set:
            if ground_truth.lower() in pred.lower():
                marked_predictions.append(f"{pred} ✓")
            else:
                marked_predictions.append(pred)
        
        # Format prediction set
        pred_text = ', '.join(marked_predictions)
        if len(pred_text) > 80:
            pred_lines = []
            current_line = ""
            for item in marked_predictions:
                if len(current_line) + len(item) + 2 > 80:
                    pred_lines.append(current_line)
                    current_line = item + ", "
                else:
                    current_line += item + ", "
            if current_line:
                pred_lines.append(current_line.rstrip(", "))
            pred_text = "\n                   ".join(pred_lines)
        
        # Format info text
        info_text = [
            f"File: {image_path.name}",
            f"Ground Truth: {ground_truth}",
            f"Method: {method} (alpha={alpha:.2f})",
            f"",
            f"Prediction Set ({set_size} classes):",
            f"{pred_text}",
            f"",
            f"Statistics:",
            f"  • Confidence: {confidence:.4f}",
            f"  • Target Coverage: {1-alpha:.2%}",
        ]
        
        props = dict(
            boxstyle='round,pad=0.8',
            facecolor='lightyellow',
            edgecolor='darkgray',
            alpha=0.95,
            linewidth=1.5
        )
        
        ax_text.text(0.05, 0.95,
                    "\n".join(info_text),
                    transform=ax_text.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=props,
                    family='monospace',
                    linespacing=1.6)
    
    return fig


class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conformal Prediction Visualization Viewer")
        self.root.geometry("1500x900")
        
        self.viz_dir = Path("MapReduceResult/Visualization")
        self.current_samples = None
        self.current_fig = None
        self.canvas_widget = None
        
        # Create UI
        self.create_ui()
        self.load_file_list()
    
    def create_ui(self):
        """Create user interface"""
        # Top control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Method selection
        ttk.Label(control_frame, text="Select Visualization File:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(control_frame, textvariable=self.file_var, state='readonly', width=50)
        self.file_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.file_combo.bind('<<ComboboxSelected>>', self.on_file_selected)
        
        ttk.Button(control_frame, text="Load", command=self.on_load_clicked).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, background="lightgray")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Canvas frame
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
    
    def load_file_list(self):
        """Load list of available visualization files"""
        if not self.viz_dir.exists():
            messagebox.showerror("Error", f"Visualization directory not found: {self.viz_dir}")
            return
        
        files = sorted(self.viz_dir.glob("*.json"))
        if not files:
            messagebox.showwarning("Warning", "No visualization files found")
            return
        
        file_names = [f.name for f in files]
        self.file_combo['values'] = file_names
        
        # Select the latest file
        if file_names:
            self.file_combo.current(len(file_names) - 1)
            self.file_var.set(file_names[-1])
    
    def on_file_selected(self, event=None):
        """Handle file selection"""
        selected_file = self.file_var.get()
        if selected_file:
            self.status_var.set(f"Selected: {selected_file}")
    
    def on_load_clicked(self):
        """Load and display selected file"""
        selected_file = self.file_var.get()
        if not selected_file:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        file_path = self.viz_dir / selected_file
        samples_data = load_visualization_file(file_path)
        
        if not samples_data:
            messagebox.showerror("Error", f"Failed to load {selected_file}")
            return
        
        self.current_samples = samples_data
        
        # Clear previous canvas
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Create new figure
        n_samples = len(samples_data)
        self.current_fig = plt.figure(figsize=(14, 4.5 * n_samples), dpi=100)
        
        visualize_samples(samples_data, self.current_fig)
        
        # Create canvas widget
        self.canvas_widget = FigureCanvasTkAgg(self.current_fig, master=self.scrollable_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update status
        first_sample = samples_data[0]
        method = first_sample['method']
        alpha = first_sample['alpha']
        self.status_var.set(f"Loaded {n_samples} samples - Method: {method}, Alpha: {alpha:.2f}")
        
        print(f"✅ Loaded {n_samples} samples from {selected_file}")


def main():
    print("\n" + "="*70)
    print("CONFORMAL PREDICTION VISUALIZATION VIEWER")
    print("="*70)
    
    root = tk.Tk()
    app = VisualizationApp(root)
    
    # Load and display the latest file by default
    root.after(500, app.on_load_clicked)
    
    root.mainloop()


if __name__ == "__main__":
    main()
