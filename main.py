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



def image_to_cloud_stipple(path, max_points=7000, iterations=5):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable : {path}")

    img = cv2.resize(img, (300, 300))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    h, w = img.shape

    darkness = (255 - img.astype(np.float32)) / 255.0
    darkness = np.clip(darkness ** 1.5, 0.01, 1.0) 

    flat = darkness.flatten()
    probs = flat / flat.sum()

  
    indices = np.random.choice(len(flat), size=max_points, replace=True, p=probs)
    pts = np.array([[idx % w, idx // w] for idx in indices], dtype=np.float32)

   
    from scipy.spatial import Voronoi

    for iteration in range(iterations):
       
        mirrored = np.concatenate([
            pts,
            pts * [-1, 1] + [2 * w, 0],   
            pts * [1, -1] + [0, 2 * h],  
            pts * [-1, -1] + [2 * w, 2 * h],
        ])

        try:
            vor = Voronoi(mirrored)
        except Exception:
            break

        new_pts = []
        for i in range(max_points):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                new_pts.append(pts[i])
                continue

            # Bounding box de la cellule
            verts = vor.vertices[region]
            x0, y0 = np.clip(verts.min(axis=0).astype(int), 0, [w-1, h-1])
            x1, y1 = np.clip(verts.max(axis=0).astype(int) + 1, 0, [w, h])

            # Centroïde pondéré par darkness dans la bbox
            if x1 <= x0 or y1 <= y0:
                new_pts.append(pts[i])
                continue

            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            gx, gy = np.meshgrid(xs, ys)
            weights = darkness[gy, gx]
            total = weights.sum()

            if total < 1e-6:
                new_pts.append(pts[i])
            else:
                cx = (gx * weights).sum() / total
                cy = (gy * weights).sum() / total
                new_pts.append([cx, cy])

        pts = np.array(new_pts, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    # --- Construire les points finaux avec radius ---
    points = []
    for px_img, py_img in pts:
        xi, yi = int(np.clip(px_img, 0, w-1)), int(np.clip(py_img, 0, h-1))
        local_d = darkness[yi, xi]
        radius = 0.5 + local_d * 3.0
        # Normalise vers [-1, 1]
        px = (px_img - w / 2) / (w / 2)
        py = -(py_img - h / 2) / (h / 2)
        points.append((px, py, radius))

    return points
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

    all_pts = [(p[0], p[1]) for p in state["cloud1"] + state["cloud2"]]
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

    # raw clouds — gros points d'abord, petits par-dessus
    if state["show_clouds"]:
        sorted_c1 = sorted(state["cloud1"], key=lambda p: p[2] if len(p) > 2 else POINT_R, reverse=True)
        sorted_c2 = sorted(state["cloud2"], key=lambda p: p[2] if len(p) > 2 else POINT_R, reverse=True)

        for p in sorted_c1:
            sx, sy = world_to_screen(p[0], p[1], scale, cx, cy)
            r = p[2] if len(p) > 2 else POINT_R
            canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=COLOR_C1, outline="")
        for p in sorted_c2:
            sx, sy = world_to_screen(p[0], p[1], scale, cx, cy)
            r = p[2] if len(p) > 2 else POINT_R
            canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=COLOR_C2, outline="")

    # interpolation — gros points d'abord, petits par-dessus
    X, Y = state["cloud1"], state["cloud2"]
    interp_points = []
    for i, j in enumerate(state["T"]):
        pt  = (1 - t) * np.array([X[i][0], X[i][1]]) + t * np.array([Y[j][0], Y[j][1]])
        r_i = X[i][2] if len(X[i]) > 2 else POINT_R
        r_j = Y[j][2] if len(Y[j]) > 2 else POINT_R
        r   = (1 - t) * r_i + t * r_j
        interp_points.append((pt[0], pt[1], r))

    interp_points.sort(key=lambda p: p[2], reverse=True)

    for pt_x, pt_y, r in interp_points:
        sx, sy = world_to_screen(pt_x, pt_y, scale, cx, cy)
        canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=COLOR_INTERP, outline="")


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
        mode = mode_var.get()  
        fn = image_to_cloud if mode == "edges" else image_to_cloud_stipple
        c1 = fn("VT.png")
        c2 = fn("NOAH.png")
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
        # Strip le radius pour les algos qui attendent du 2D
        X2D = [(p[0], p[1]) for p in state["cloud1"]]
        Y2D = [(p[0], p[1]) for p in state["cloud2"]]
        T = bijection_tournament(X2D, Y2D, niveau)
        state["T"] = T
        root.after(0, lambda: (hide_overlay(), slider.set(0), redraw(0.0),
                               status_label.config(text="")))
    show_overlay(f"Calcul bijection (niveau {niveau})…")
    threading.Thread(target=task, daemon=True).start()


def compute_cost():
    if not state["T"]:
        status_label.config(text="Pas de bijection calculée.")
        return
    X2D = [(p[0], p[1]) for p in state["cloud1"]]
    Y2D = [(p[0], p[1]) for p in state["cloud2"]]
    cost = matching_cost(X2D, Y2D, state["T"])
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


mode_var = tk.StringVar(value="edges")
tk.Radiobutton(row1, text="Edges", variable=mode_var, value="edges",
               bg=COLOR_BG, fg="#aaaaaa", selectcolor=COLOR_BTN,
               activebackground=COLOR_BG, font=("Courier", 9)).pack(side=tk.LEFT)
tk.Radiobutton(row1, text="Stipple", variable=mode_var, value="stipple",
               bg=COLOR_BG, fg="#aaaaaa", selectcolor=COLOR_BTN,
               activebackground=COLOR_BG, font=("Courier", 9)).pack(side=tk.LEFT, padx=(0, 8))

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


