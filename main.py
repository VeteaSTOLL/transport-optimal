import tkinter as tk
import threading
import random
from clouds import *
import cv2

# ── Constants ────────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 600, 600
MARGIN      = 20
POINT_R     = 2
GRID_LINES  = 5

COLOR_BG      = "#0f0f0f"
COLOR_GRID    = "#2a2a2a"
COLOR_AXIS    = "#444444"
COLOR_C1      = "#4fc3f7"
COLOR_C2      = "#ff7043"
COLOR_INTERP  = "#00e676"
COLOR_BTN     = "#1e1e1e"
COLOR_BTN_ACT = "#333333"

BTN_OPTS = dict(bg=COLOR_BTN, fg="#ffffff",
                activebackground=COLOR_BTN_ACT, activeforeground="#ffffff",
                relief="flat", padx=12, pady=6, cursor="hand2")

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "cloud1": [],
    "cloud2": [],
    "T": [],
    "show_clouds": False,
}


# ── Point-cloud helpers ───────────────────────────────────────────────────────
def image_to_cloud(path, max_points=2000):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable : {path}")
    edges = cv2.Canny(img, 100, 200)
    h, w  = edges.shape
    points = [
        ((x - w/2) / (w/2), -(y - h/2) / (h/2))
        for y in range(h) for x in range(w) if edges[y, x]
    ]
    if len(points) > max_points:
        points = random.sample(points, max_points)
    return points


def make_cloud():
    mu    = np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)])
    theta = random.uniform(0, np.pi)
    a     = random.uniform(1.5, 3.5)
    b     = random.uniform(0.5, min(a - 0.3, 1.8))
    u1    = a * np.array([ np.cos(theta),  np.sin(theta)])
    u2    = b * np.array([-np.sin(theta),  np.cos(theta)])
    n     = random.randint(150, 400)
    return generate_cloud_ellipse(n, mu, u1, u2)


def matching_cost(X, Y, T):
    total = sum(
        ((X[i][0] - Y[j][0])**2 + (X[i][1] - Y[j][1])**2) ** 0.5
        for i, j in enumerate(T)
    )
    return total / len(T)


# ── Drawing ───────────────────────────────────────────────────────────────────
def world_to_screen(px, py, scale, cx, cy):
    return cx + px * scale, cy - py * scale


def redraw(t=0.0):
    canvas.delete("all")
    cx, cy  = CANVAS_W // 2, CANVAS_H // 2
    all_pts = state["cloud1"] + state["cloud2"]
    if not all_pts:
        return
    max_val = max(abs(v) for p in all_pts for v in p) * 1.15 or 1
    scale   = (min(CANVAS_W - 2*MARGIN, CANVAS_H - 2*MARGIN) / 2) / max_val

    # grid
    step = max_val / GRID_LINES
    val  = step
    while val <= max_val * 1.05:
        for sign in (1, -1):
            sx, _ = world_to_screen(sign * val, 0, scale, cx, cy)
            canvas.create_line(sx, MARGIN, sx, CANVAS_H - MARGIN, fill=COLOR_GRID)
            _, sy = world_to_screen(0, sign * val, scale, cx, cy)
            canvas.create_line(MARGIN, sy, CANVAS_W - MARGIN, sy, fill=COLOR_GRID)
        val += step
    canvas.create_line(cx, MARGIN, cx, CANVAS_H - MARGIN, fill=COLOR_AXIS)
    canvas.create_line(MARGIN, cy, CANVAS_W - MARGIN, cy, fill=COLOR_AXIS)

    # raw clouds
    if state["show_clouds"]:
        for px, py in state["cloud1"]:
            sx, sy = world_to_screen(px, py, scale, cx, cy)
            canvas.create_oval(sx-POINT_R, sy-POINT_R, sx+POINT_R, sy+POINT_R,
                               fill=COLOR_C1, outline="")
        for px, py in state["cloud2"]:
            sx, sy = world_to_screen(px, py, scale, cx, cy)
            canvas.create_oval(sx-POINT_R, sy-POINT_R, sx+POINT_R, sy+POINT_R,
                               fill=COLOR_C2, outline="")

    # interpolated bijection
    X, Y = state["cloud1"], state["cloud2"]
    for i, j in enumerate(state["T"]):
        pt = (1 - t) * np.array(X[i]) + t * np.array(Y[j])
        sx, sy = world_to_screen(pt[0], pt[1], scale, cx, cy)
        canvas.create_oval(sx-POINT_R, sy-POINT_R, sx+POINT_R, sy+POINT_R,
                           fill=COLOR_INTERP, outline="")


def on_slider(val):
    redraw(float(val) / 100)


# ── Loading overlay ───────────────────────────────────────────────────────────
def show_overlay(msg="Calcul en cours…"):
    overlay_label.config(text=msg)
    overlay_frame.place(relx=0.5, rely=0.5, anchor="center")
    root.update_idletasks()


def hide_overlay():
    overlay_frame.place_forget()


# ── Actions ───────────────────────────────────────────────────────────────────
def load_images():
    def task():
        c1 = image_to_cloud("VT.png")
        c2 = image_to_cloud("NOAH.png")
        n  = min(len(c1), len(c2))
        state["cloud1"] = c1[:n]
        state["cloud2"] = c2[:n]
        state["T"] = []
        root.after(0, lambda: (hide_overlay(), slider.set(0), redraw(0.0)))
    show_overlay("Chargement des images…")
    threading.Thread(target=task, daemon=True).start()


def load_random():
    c1 = make_cloud()
    c2 = make_cloud()
    n  = min(len(c1), len(c2))
    state["cloud1"] = c1[:n]
    state["cloud2"] = c2[:n]
    state["T"] = []
    slider.set(0)
    redraw(0.0)


def compute_bijection():
    if not state["cloud1"] or not state["cloud2"]:
        status_label.config(text="Chargez d'abord des nuages.")
        return

    niveau = niveau_var.get()

    def task():
        T = bijection_tournament(state["cloud1"], state["cloud2"], niveau)
        state["T"] = T
        root.after(0, lambda: (hide_overlay(), slider.set(0), redraw(0.0),
                               status_label.config(text="")))
    show_overlay(f"Calcul bijection (niveau {niveau})…")
    threading.Thread(target=task, daemon=True).start()


def compute_cost():
    if not state["T"]:
        status_label.config(text="Pas de bijection calculée.")
        return
    cost = matching_cost(state["cloud1"], state["cloud2"], state["T"])
    status_label.config(text=f"Coût total : {cost:.4f}")


def toggle_clouds():
    state["show_clouds"] = not state["show_clouds"]
    redraw(slider.get() / 100)


# ── UI layout ─────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Nuages de points 2D")
root.configure(bg=COLOR_BG)
root.resizable(False, False)

frame = tk.Frame(root, bg=COLOR_BG)
frame.pack(padx=10, pady=10)

# Canvas
canvas_frame = tk.Frame(frame, bg=COLOR_BG)
canvas_frame.grid(row=0, column=0)

canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H,
                   bg=COLOR_BG, highlightthickness=0)
canvas.pack()

# Loading overlay (shown on top of the canvas when computing)
overlay_frame = tk.Frame(canvas_frame, bg="#1a1a2e", padx=20, pady=14)
overlay_label = tk.Label(overlay_frame, text="", bg="#1a1a2e", fg="#00e676",
                         font=("Courier", 13, "bold"))
overlay_label.pack()

# Slider
slider = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL,
                  length=CANVAS_W, command=on_slider,
                  bg=COLOR_BG, fg="#ffffff", troughcolor="#333333",
                  highlightthickness=0)
slider.grid(row=1, column=0, pady=(6, 0))


def section_label(parent, text):
    tk.Label(parent, text=text, bg=COLOR_BG, fg="#555555",
             font=("Courier", 8)).pack(side=tk.LEFT, padx=(0, 8))


# ── Ligne 1 : Génération de points ───────────────────────────────────────────
row1 = tk.Frame(frame, bg=COLOR_BG)
row1.grid(row=2, column=0, pady=(12, 0), sticky="w")
section_label(row1, "Génération :")

tk.Button(row1, text="Images",  command=load_images,  **BTN_OPTS).pack(side=tk.LEFT, padx=4)
tk.Button(row1, text="Random",  command=load_random,  **BTN_OPTS).pack(side=tk.LEFT, padx=4)

# ── Ligne 2 : Calcul de bijection ────────────────────────────────────────────
row2 = tk.Frame(frame, bg=COLOR_BG)
row2.grid(row=3, column=0, pady=(6, 0), sticky="w")
section_label(row2, "Bijection  :")

tk.Label(row2, text="Niveau", bg=COLOR_BG, fg="#aaaaaa",
         font=("Courier", 9)).pack(side=tk.LEFT)

niveau_var = tk.IntVar(value=3)
niveau_spin = tk.Spinbox(row2, from_=0, to=10, textvariable=niveau_var, width=4,
                         bg=COLOR_BTN, fg="#ffffff", buttonbackground=COLOR_BTN_ACT,
                         relief="flat", font=("Courier", 10))
niveau_spin.pack(side=tk.LEFT, padx=(4, 10))

tk.Button(row2, text="Calculer", command=compute_bijection,
          bg=COLOR_BTN, fg="#4fc3f7",
          activebackground=COLOR_BTN_ACT, activeforeground="#4fc3f7",
          relief="flat", padx=12, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=4)

# ── Ligne 3 : Utilitaires ────────────────────────────────────────────────────
row3 = tk.Frame(frame, bg=COLOR_BG)
row3.grid(row=4, column=0, pady=(6, 0), sticky="w")
section_label(row3, "Utilitaires:")

tk.Button(row3, text="Calcul coût",         command=compute_cost,   **BTN_OPTS).pack(side=tk.LEFT, padx=4)
tk.Button(row3, text="Afficher/cacher nuages", command=toggle_clouds, **BTN_OPTS).pack(side=tk.LEFT, padx=4)

# Status bar
status_label = tk.Label(frame, text="", bg=COLOR_BG, fg="#00e676",
                         font=("Courier", 10))
status_label.grid(row=5, column=0, pady=(8, 4))

root.mainloop()
