import tkinter as tk
import random
from clouds import *
import cv2
import random


CANVAS_W, CANVAS_H = 600, 600
MARGIN     = 20
POINT_R    = 2
GRID_LINES = 5
COLOR_BG   = "#0f0f0f"
COLOR_GRID = "#2a2a2a"
COLOR_AXIS = "#444444"
COLOR_C1   = "#4fc3f7"
COLOR_C2   = "#ff7043"
COLOR_E1   = "#ffffff"
COLOR_E2   = "#ffcc00"
COLOR_INTERP = "#00e676"



state = {}
current_mode = "image"

def image_to_cloud(path, max_points=2000):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image introuvable")

    edges = cv2.Canny(img, 100, 200)

    points = []

    h, w = edges.shape

    for y in range(h):
        for x in range(w):
            if edges[y, x] != 0:
               
                px = (x - w / 2) / (w / 2)
                py = (y - h / 2) / (h / 2)
                py = -py  
                points.append((px, py))

    if len(points) > max_points:
        points = random.sample(points, max_points)

    return points


def make_cloud(n=None):
    mu    = np.array([random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)])
    theta = random.uniform(0, np.pi)
    a     = random.uniform(1.5, 3.5)
    b     = random.uniform(0.5, min(a - 0.3, 1.8))
    u1    = a * np.array([ np.cos(theta),  np.sin(theta)])
    u2    = b * np.array([-np.sin(theta),  np.cos(theta)])
    if n is None:
        n = random.randint(150, 400)
    cloud = generate_cloud_ellipse(n, mu, u1, u2)
    mu_c  = np.array(mean_cloud(cloud))
    C     = variance_cloud(cloud, mu_c)
    w, V  = np.linalg.eigh(C)
    uc1   = np.sqrt(abs(w[0])) * V[:, 0]
    uc2   = np.sqrt(abs(w[1])) * V[:, 1]
    return cloud, mu_c, uc1, uc2, C

def reset(mode=None):
    global current_mode
    state["show_cloud"] = False

    if mode is not None:
        current_mode = mode

    if current_mode == "image":
        cloud1 = image_to_cloud("NOAMIGO.png")
        cloud2 = image_to_cloud("AMIGO.jpg")
    else:
        n = random.randint(150, 400)
        cloud1 = make_cloud(n)[0]
        cloud2 = make_cloud(n)[0]

    n = min(len(cloud1), len(cloud2))
    cloud1 = cloud1[:n]
    cloud2 = cloud2[:n]

    c1 = list(cloud1)
    c2 = list(cloud2)

    T_dict = {}
    BSP_matching(c1, c2, 0, n, T_dict)

    state["cloud1"] = cloud1
    state["cloud2"] = cloud2
    state["T_dict"] = T_dict

    slider.set(0)
    redraw(0.0)

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

def redraw(t=0.0):
    canvas.delete("all")
    w, h   = CANVAS_W, CANVAS_H
    cx, cy = w // 2, h // 2
    all_pts = state['cloud1'] + state['cloud2']
    max_val = max(abs(v) for p in all_pts for v in p) * 1.15 or 1
    scale   = (min(w - 2*MARGIN, h - 2*MARGIN) / 2) / max_val

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
    canvas.create_line(MARGIN, cy, w - MARGIN, cy,  fill=COLOR_AXIS)
    if state.get("show_cloud", True):
        for px, py in state['cloud1']:
            sx, sy = world_to_screen(px, py, scale, cx, cy)
            canvas.create_oval(sx-POINT_R, sy-POINT_R, sx+POINT_R, sy+POINT_R, fill=COLOR_C1, outline="")
    if state.get("show_cloud", True):
        for px, py in state['cloud2']:
            sx, sy = world_to_screen(px, py, scale, cx, cy)
            canvas.create_oval(
                sx-POINT_R, sy-POINT_R,
                sx+POINT_R, sy+POINT_R,
                fill=COLOR_C2, outline=""
            )

    # points verts interpolés via BSP matching
    for src, dst in state['T_dict'].items():
        pt = (1 - t) * np.array(src) + t * np.array(dst)
        sx, sy = world_to_screen(pt[0], pt[1], scale, cx, cy)
        canvas.create_oval(sx-POINT_R, sy-POINT_R, sx+POINT_R, sy+POINT_R, fill=COLOR_INTERP, outline="")

def on_slider(val):
    redraw(float(val) / 100)


def toggle_cloud2():
    state["show_cloud"] = not state.get("show_cloud", True)
    redraw(slider.get() / 100)

root = tk.Tk()
root.title("Nuages de points 2D")
root.configure(bg=COLOR_BG)
root.resizable(False, False)

frame = tk.Frame(root, bg=COLOR_BG)
frame.pack(padx=10, pady=10)

canvas = tk.Canvas(frame, width=CANVAS_W, height=CANVAS_H, bg=COLOR_BG, highlightthickness=0)
canvas.grid(row=0, column=0)

slider = tk.Scale(
    frame, from_=0, to=100, orient=tk.HORIZONTAL,
    length=CANVAS_W,
    command=on_slider,
    bg=COLOR_BG, fg="#ffffff", troughcolor="#333333", highlightthickness=0
)
slider.grid(row=1, column=0, pady=(6, 0))

btn_reset = tk.Button(
    frame, text="Reset", command=reset,
    bg="#1e1e1e", fg="#ffffff", activebackground="#333333", activeforeground="#ffffff",
    relief="flat", padx=12, pady=6, cursor="hand2"
)

btn_toggle = tk.Button(
    frame,
    text="Show/Hide Cloud ",
    command=toggle_cloud2,
    bg="#1e1e1e",
    fg="#ffffff",
    activebackground="#333333",
    activeforeground="#ffffff",
    relief="flat",
    padx=12,
    pady=6,
    cursor="hand2"
)



btn_img = tk.Button(
    frame,
    text="Images",
    command=lambda: reset("image"),
    bg="#1e1e1e",
    fg="#ffffff",
    activebackground="#333333",
    activeforeground="#ffffff",
    relief="flat",
    padx=12,
    pady=6,
    cursor="hand2"
)

btn_rand = tk.Button(
    frame,
    text="Random",
    command=lambda: reset("random"),
    bg="#1e1e1e",
    fg="#ffffff",
    activebackground="#333333",
    activeforeground="#ffffff",
    relief="flat",
    padx=12,
    pady=6,
    cursor="hand2"
)

btn_reset.grid(row=2, column=0, pady=(8, 0))
btn_img.grid(row=3, column=0, pady=(4, 0))
btn_rand.grid(row=4, column=0, pady=(4, 0))
btn_toggle.grid(row=5, column=0, pady=(4, 0))


reset("image")

root.mainloop()
