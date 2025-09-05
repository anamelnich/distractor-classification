# demogrphics.py
# A local GUI for demographic and screening questions, saving responses as two-column text.

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os, sys


# ==== Accept subjectID and folder path ====
if len(sys.argv) < 3:
    print("Usage: python demographics.py <subjectID> <output_folder>")
    sys.exit(1)

subject_id = sys.argv[1]      # e.g., "subject0"
output_dir = sys.argv[2]      # e.g., "./data/e0_20250905"
os.makedirs(output_dir, exist_ok=True)

# Output filename format
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"demographics_{subject_id}_{timestamp}.txt")

# Create main window
tk_root = tk.Tk()
tk_root.title("Demographics Questionnaire")
tk_root.attributes('-fullscreen', True)  # Make it fullscreen

# Create a canvas with scrollbar
canvas = tk.Canvas(tk_root)
scrollbar = tk.Scrollbar(tk_root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Mouse wheel scrolling
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

# Custom confirmation dialog with "Save" and "Return" buttons
def confirm_dialog(title, message):
    dlg = tk.Toplevel(tk_root)
    dlg.title(title)
    dlg.grab_set()
    tk.Label(dlg, text=message, wraplength=400, justify='left', font=(None, 12)).pack(padx=20, pady=10)
    result = {'confirm': False}
    def on_save():
        result['confirm'] = True
        dlg.destroy()
    def on_return():
        dlg.destroy()
    btn_frame = tk.Frame(dlg)
    btn_frame.pack(pady=(0,10))
    tk.Button(btn_frame, text="Save", width=10, command=on_save).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Return", width=10, command=on_return).pack(side='left', padx=10)
    dlg.protocol("WM_DELETE_WINDOW", on_return)
    dlg.transient(tk_root)
    dlg.wait_window()
    return result['confirm']

# Q1: First name
tk.Label(scrollable_frame, text="1. First name:", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
first_name_entry = tk.Entry(scrollable_frame, font=(None, 14))
first_name_entry.pack(fill='x', padx=10)

# Q2: Last name
tk.Label(scrollable_frame, text="2. Last name:", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
last_name_entry = tk.Entry(scrollable_frame, font=(None, 14))
last_name_entry.pack(fill='x', padx=10)

# Q3: Age (18-99 numeric only)
tk.Label(scrollable_frame, text="3. How old are you? (18-99)", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
age_var = tk.StringVar()
vcmd = (tk_root.register(lambda P: (P.isdigit() and len(P) <= 2) or P == ""), '%P')
age_entry = tk.Entry(scrollable_frame, textvariable=age_var, validate='key', validatecommand=vcmd, font=(None, 14))
age_entry.pack(fill='x', padx=10)

# Q4: Biological sex assigned at birth
tk.Label(scrollable_frame, text="4. What was your biological sex assigned at birth?", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
sex_var = tk.StringVar()
for opt in ["Female", "Male", "Intersex", "None of these describe me", "Prefer not to answer"]:
    tk.Radiobutton(scrollable_frame, text=opt, variable=sex_var, value=opt, font=(None, 14)).pack(anchor='w', padx=20)

# Q5: Race
tk.Label(scrollable_frame, text="5. Which of the following best describes your race?", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
race_var = tk.StringVar()
race_opts = [
    "American Indian or Alaska Native",
    "Asian",
    "Black or African American",
    "Native Hawaiian or Other Pacific Islander",
    "White or Caucasian",
    "More than 1 race or multi-racial",
    "Other",
    "Prefer not to answer"
]
for opt in race_opts:
    tk.Radiobutton(scrollable_frame, text=opt, variable=race_var, value=opt, font=(None, 14)).pack(anchor='w', padx=20)


# Q6: Hispanic/Latino
tk.Label(scrollable_frame, text="6. Do you consider yourself to be Hispanic or Latino?", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
hisp_var = tk.StringVar()
for opt in ["YES", "NO", "Prefer not to answer"]:
    tk.Radiobutton(scrollable_frame, text=opt, variable=hisp_var, value=opt, font=(None, 14)).pack(anchor='w', padx=20)

# Q7: Vision
tk.Label(scrollable_frame, text="7. Do you have normal or corrected-to-normal vision?", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
tk.Label(scrollable_frame, text="(corrected-to-normal means use of corrective lenses like glasses or contacts)", font=(None, 12, 'italic')).pack(anchor='w', padx=20)
vision_var = tk.StringVar()
for opt in ["YES", "NO"]:
    tk.Radiobutton(scrollable_frame, text=opt, variable=vision_var, value=opt, font=(None, 14)).pack(anchor='w', padx=20)

# Q8: Neurological history
tk.Label(scrollable_frame, text="8. History of neurological disease or condition? (e.g., stroke, Alzheimer’s, epilepsy, etc.)", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
tk.Label(scrollable_frame, text="(e.g., stroke, Alzheimer's disease, Parkinson's disease, epilepsy, multiple sclerosis, traumatic brain injuries)", font=(None, 12, 'italic')).pack(anchor='w', padx=20)
neuro_var = tk.StringVar()
for opt in ["YES", "NO"]:
    tk.Radiobutton(scrollable_frame, text=opt, variable=neuro_var, value=opt, font=(None, 14)).pack(anchor='w', padx=20)

# Q9: Email
tk.Label(scrollable_frame, text="9. Email address:", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
email_entry = tk.Entry(scrollable_frame, font=(None, 14))
email_entry.pack(fill='x', padx=10)

# Q10: Phone number
tk.Label(scrollable_frame, text="10. Phone number:", font=(None, 16)).pack(anchor='w', padx=10, pady=(10,0))
phone_entry = tk.Entry(scrollable_frame, font=(None, 14))
phone_entry.pack(fill='x', padx=10)

# Save responses
def save_answers():
    missing = []
    if not first_name_entry.get().strip(): missing.append("First name")
    if not last_name_entry.get().strip(): missing.append("Last name")
    age = age_var.get()
    if not age or not age.isdigit(): missing.append("Age")
    if not sex_var.get(): missing.append("Biological sex")
    if not race_var.get(): missing.append("Race")
    if not hisp_var.get(): missing.append("Hispanic/Latino status")
    if not vision_var.get(): missing.append("Vision status")
    if not neuro_var.get(): missing.append("Neurological history")
    if not email_entry.get().strip(): missing.append("Email address")
    if missing:
        messagebox.showerror("Required Fields Missing",
            "Please complete the following required fields:\n- " + "\n- ".join(missing)
        )
        return
    # Validate age
    age = age_var.get()
    if not age.isdigit() or not (18 <= int(age) <= 99):
        messagebox.showerror("Invalid Age", "Please enter a number between 18 and 99.")
        return
    # Vision warning
    if vision_var.get() == "NO":
        if not confirm_dialog(
            "Vision Warning",
            "Please wait for the experimenter."
        ):
            return
    # Neuro history warning
    if neuro_var.get() == "YES":
        if not confirm_dialog(
            "Neurological History Warning",
            "Please wait for the experimenter."
        ):
            return
    # Compile answers
    resp = [
        first_name_entry.get().strip(),
        last_name_entry.get().strip(),
        age,
        sex_var.get(),
        race_var.get(),
        hisp_var.get(),
        vision_var.get(),
        neuro_var.get(),
        email_entry.get().strip(),
        phone_entry.get().strip()
    ]
    # Save file
    try:
        with open(output_file, 'w') as f:
            for i, ans in enumerate(resp, start=1):
                f.write(f"{i}\t{ans}\n")
        messagebox.showinfo("Saved", f"Responses saved to {output_file}")
        tk_root.destroy()
    except Exception as e:
        messagebox.showerror("Error", f"Could not save responses:\n{e}")


# Save button
save_btn = tk.Button(scrollable_frame, text="Save", command=save_answers, font=(None, 16))
save_btn.pack(pady=20)

# Start GUI
tk_root.mainloop()
