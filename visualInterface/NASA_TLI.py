# NASA_TLI.py
# A local GUI showing all NASA-TLX questions at once with 1–20 sliders,
# saving responses as two-column text.

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os, sys

# ==== Accept subjectID and folder path ====
if len(sys.argv) < 3:
    print("Usage: python NASA_TLI.py <subjectID> <output_folder>")
    sys.exit(1)

subject_id = sys.argv[1]      # e.g., "subject0"
output_dir = sys.argv[2]      # e.g., "./data/e0_20250905"
os.makedirs(output_dir, exist_ok=True)

# Output filename format
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"NASAtli_{subject_id}_{timestamp}.txt")

# Define the questions and their endpoint labels
questions = [
    {"text": "Mental Demand: How mentally demanding was the task?",             "min_label": "1 = Very Low", "max_label": "20 = Very High"},
    {"text": "Physical Demand: How physically demanding was the task?",            "min_label": "1 = Very Low", "max_label": "20 = Very High"},
    {"text": "Temporal Demand: How hurried or rushed was the pace of the task?",     "min_label": "1 = Very Low", "max_label": "20 = Very High"},
    {"text": "Performance: How successful were you in accomplishing what you were asked to do?", "min_label": "1 = Perfect",  "max_label": "20 = Failure"},
    {"text": "Effort: How hard did you have to work to accomplish your level of performance?", "min_label": "1 = Very Low", "max_label": "20 = Very High"},
    {"text": "Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?",    "min_label": "1 = Very Low", "max_label": "20 = Very High"}
]

# Create main window
tk_root = tk.Tk()
tk_root.title("Self-Report Questionnaire")
tk_root.geometry("700x750")

# Dictionary to hold each slider widget
scales = {}

# Build all questions at once
for idx, q in enumerate(questions):
    frame = tk.Frame(tk_root)
    frame.pack(fill='x', pady=20, padx=10)  # Increased pady from 5 to 20 for more spacing

    # Question label
    lbl = tk.Label(frame,
                   text=f"{idx+1}. {q['text']}",
                   wraplength=650,
                   justify='left',
                   font=(None, 16))
    lbl.pack(anchor='w')

    # Slider with endpoint labels
    slider_frame = tk.Frame(frame)
    slider_frame.pack(fill='x', pady=(10,0))  # Increased spacing between question text and slider

    min_lbl = tk.Label(slider_frame, text=q['min_label'], font=(None, 12))
    min_lbl.pack(side='left')

    scale = tk.Scale(slider_frame,
                     from_=1,
                     to=20,
                     orient='horizontal',
                     length=400)
    scale.set(10)
    scale.pack(side='left', padx=10)

    max_lbl = tk.Label(slider_frame, text=q['max_label'], font=(None, 12))
    max_lbl.pack(side='left')

    scales[idx] = scale

# Save responses to two-column text file
def save_answers(output_file):
    try:
        with open(output_file, 'w') as f:
            for i in range(len(questions)):
                f.write(f"{i+1}\t{scales[i].get()}\n")
        messagebox.showinfo("Saved", f"Responses saved to {output_file}")
        tk_root.destroy()
    except Exception as e:
        messagebox.showerror("Error", f"Could not save responses:\n{e}")

# Save button
save_btn = tk.Button(tk_root, text="Save",  command=lambda: save_answers(output_file))
save_btn.pack(pady=15)

# Start GUI loop
tk_root.mainloop()

