import tkinter as tk
import random
from clouds import *

CANVAS_W, CANVAS_H = 500, 500
MARGIN     = 20
POINT_R    = 2
GRID_LINES = 5
COLOR_BG   = "#0f0f0f"
COLOR_GRID = "#2a2a2a"
COLOR_AXIS = "#444444"
COLOR_C1   = "#4fc3f7"
COLOR_TRUE = "#e60000"
COLOR_CALC = "#ffffff"

def generate_data():
    mu_true  = np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)])
    theta    = random.uniform(0, np.pi)
    a        = random.uniform(1.5, 3.5)
    b        = random.uniform(0.5, min(a - 0.3, 1.8))
    u1_true  = a * np.array([ np.cos(theta),  np.sin(theta)])
    u2_true  = b * np.array([-np.sin(theta),  np.cos(theta)])

    n1 = random.randint(150, 500)
    cloud = generate_cloud_ellipse(n1, mu_true, u1_true, u2_true)

    mu = mean_cloud(cloud)
    C  = variance_cloud(cloud, mu)
    valeurs, vecteurs = np.linalg.eig(C)
    u1 = np.sqrt(valeurs[0]) * vecteurs[:, 0]
    u2 = np.sqrt(valeurs[1]) * vecteurs[:, 1]

    return cloud, mu_true, u1_true, u2_true, mu, u1, u2

def world_to_screen(px, py, scale, cx, cy):
    return cx + px * scale, cy - py * scale

def draw_ellipse(canvas, e_mu, e_u1, e_u2, color, scale, cx, cy):
    t_vals = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    coords = []
    for t in t_vals:
        px = e_mu[0] + np.cos(t) * e_u1[0] + np.sin(t) * e_u2[0]
        py = e_mu[1] + np.cos(t) * e_u1[1] + np.sin(t) * e_u2[1]
        sx, sy = world_to_screen(px, py, scale, cx, cy)
        coords.extend([sx, sy])
    canvas.create_polygon(*coords, outline=color, fill="", width=2, smooth=True)

def draw_canvas(canvas, points, color, ellipses=None):
    canvas.delete("all")
    w, h = CANVAS_W, CANVAS_H
    cx, cy = w // 2, h // 2
    max_val = max(abs(v) for p in points for v in p) * 1.15 or 1
    scale = (min(w - 2 * MARGIN, h - 2 * MARGIN) / 2) / max_val

    step = max_val / GRID_LINES
    val  = step
    while val <= max_val * 1.05:
        for sign in (1, -1):
            sx, _ = world_to_screen(sign * val, 0, scale, cx, cy)
            canvas.create_line(sx, MARGIN, sx, h - MARGIN, fill=COLOR_GRID)
            _, sy = world_to_screen(0, sign * val, scale, cx, cy)
            canvas.create_line(MARGIN, sy, w - MARGIN, sy, fill=COLOR_GRID)
        val += step

    canvas.create_line(cx, MARGIN, cx, h - MARGIN, fill=COLOR_AXIS)
    canvas.create_line(MARGIN, cy, w - MARGIN, cy, fill=COLOR_AXIS)

    for px, py in points:
        sx, sy = world_to_screen(px, py, scale, cx, cy)
        canvas.create_oval(sx - POINT_R, sy - POINT_R,
                           sx + POINT_R, sy + POINT_R,
                           fill=color, outline="")

    if ellipses:
        for (e_mu, e_u1, e_u2, e_color) in ellipses:
            draw_ellipse(canvas, e_mu, e_u1, e_u2, e_color, scale, cx, cy)

def reset():
    cloud, mu_true, u1_true, u2_true, mu, u1, u2 = generate_data()
    draw_canvas(c1, cloud, COLOR_C1, ellipses=[
        (mu_true, u1_true, u2_true, COLOR_TRUE),
        (mu,      u1,      u2,      COLOR_CALC),
    ])

root = tk.Tk()
root.title("Nuages de points 2D")
root.configure(bg=COLOR_BG)
root.resizable(False, False)

frame = tk.Frame(root, bg=COLOR_BG)
frame.pack(padx=10, pady=10)

c1 = tk.Canvas(frame, width=CANVAS_W, height=CANVAS_H, bg=COLOR_BG, highlightthickness=0)
c1.grid(row=0, column=0, padx=(0, 8))

btn_reset = tk.Button(
    frame, text="Reset", command=reset,
    bg="#1e1e1e", fg="#ffffff", activebackground="#333333", activeforeground="#ffffff",
    relief="flat", padx=12, pady=6, cursor="hand2"
)
btn_reset.grid(row=1, column=0, pady=(8, 0))

reset()

root.mainloop()
