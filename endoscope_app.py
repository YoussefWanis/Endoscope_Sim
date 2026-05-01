"""
endoscope_app.py  —  C3VD Colonoscopy Endoscope Simulator
SBE3220 Task 02  |  Intelligent Endoscopic Assistance System
=============================================================

Dataset:  C3VD — Colonoscopy 3D Video Dataset (Johns Hopkins, CC BY-NC-SA 4.0)
          https://durrlab.github.io/C3VD/

Run:
    python endoscope_app.py

If no C3VD data is found it runs in DEMO mode with synthetic frames
and prints instructions on where to install the dataset.

Controls (keyboard):
    Arrow keys      — pan camera tip
    , / .           — previous / next frame
    A / D           — previous / next sequence
    [ / ]           — previous / next sequence
    Space           — capture frame
    1-5             — toggle processing modes
    +/-             — zoom in / out
    Escape          — quit
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os, sys, math, time, random, json, threading, struct, zlib, re
import numpy as np

try:
    from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    print("ERROR: Pillow not installed. Run:  pip install Pillow")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from skimage.feature import local_binary_pattern
    from skimage.measure import label, regionprops
    HAS_SKI = True
except ImportError:
    HAS_SKI = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', 'c3vd')
OUT_DIR  = os.path.join(BASE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
BG      = '#06080c'
PANEL   = '#0b0f16'
CARD    = '#111820'
BORDER  = '#1c2535'
ACC     = '#4d9de0'
ACC2    = '#3cb371'
ACC3    = '#e05c5c'
HUD_G   = '#00e87a'
TXT     = '#d0dcea'
TXT2    = '#4a6480'
FNT     = ('Consolas', 10)
FNT_S   = ('Consolas', 9)
FNT_L   = ('Consolas', 13, 'bold')
FNT_H   = ('Consolas', 11, 'bold')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET LAYER                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MANUAL_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║            C3VD DATASET — WHERE TO PUT YOUR FILES                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  No C3VD data was found. The app is running in DEMO mode.                   ║
║                                                                              ║
║  To use real colonoscopy video frames:                                       ║
║                                                                              ║
║  OPTION A  — Auto-install (requires internet)                               ║
║    Run:  python install_dataset.py                                           ║
║    Then: python endoscope_app.py                                             ║
║                                                                              ║
║  OPTION B  — Manual install                                                  ║
║    1. Download a sequence zip from:                                          ║
║         https://durrlab.github.io/C3VD/                                      ║
║       Recommended (smallest):                                                ║
║         trans_t1_a.zip   (~0.59 GB, 61 frames)                              ║
║         trans_t2_b.zip   (~0.97 GB, 103 frames)                             ║
║                                                                              ║
║    2. Extract the zip — you'll get a folder containing images, depth, etc.  ║
║                                                                              ║
║    3. Place it at exactly this path:                                         ║
║         {DATA_DIR}                                                           ║
║       So the final structure is:                                             ║
║         data/c3vd/trans_t1_a/0001.png                                       ║
║         data/c3vd/trans_t1_a/depth/0001_depth.tiff                          ║
║         data/c3vd/trans_t1_a/pose.txt                                       ║
║                                                                              ║
║    4. Re-launch:  python endoscope_app.py                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".format(DATA_DIR=DATA_DIR)


class C3VDDataset:
    """Loads real C3VD frames from disk, or generates synthetic ones."""

    def __init__(self):
        self.sequences  = []
        self.seq_idx    = 0
        self.frame_idx  = 0
        self._cache     = {}
        self._scan()

    def _scan(self):
        self.sequences = []
        if not os.path.isdir(DATA_DIR):
            return
        for name in sorted(os.listdir(DATA_DIR)):
            seq_dir = os.path.join(DATA_DIR, name)
            if not os.path.isdir(seq_dir):
                continue
            
            rgb_dir = os.path.join(seq_dir, 'rgb')
            if not os.path.isdir(rgb_dir):
                rgb_dir = seq_dir
                
            frames = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Natural sort so '9.png' comes before '10.png'
            frames.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])
            if not frames:
                continue
            depth_dir = os.path.join(seq_dir, 'depth')
            pose_file = os.path.join(seq_dir, 'pose.txt')
            poses = self._load_poses(pose_file) if os.path.exists(pose_file) else []
            self.sequences.append({
                'name':      name,
                'rgb_dir':   rgb_dir,
                'depth_dir': depth_dir if os.path.isdir(depth_dir) else None,
                'frames':    frames,
                'poses':     poses,
            })

    def _load_poses(self, path):
        poses = []
        try:
            with open(path) as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    if len(vals) == 16:
                        poses.append(np.array(vals).reshape(4, 4))
        except Exception:
            pass
        return poses

    @property
    def has_real_data(self):
        return len(self.sequences) > 0

    @property
    def current_seq(self):
        if not self.sequences:
            return None
        return self.sequences[self.seq_idx % len(self.sequences)]

    @property
    def n_frames(self):
        seq = self.current_seq
        return len(seq['frames']) if seq else 0

    @property
    def current_pose(self):
        seq = self.current_seq
        if seq and seq['poses'] and self.frame_idx < len(seq['poses']):
            return seq['poses'][self.frame_idx]
        return None

    def get_frame(self, seq_idx=None, frame_idx=None):
        """Return (rgb_pil, depth_pil_or_None)."""
        si = seq_idx if seq_idx is not None else self.seq_idx
        fi = frame_idx if frame_idx is not None else self.frame_idx
        if not self.sequences:
            return None, None
        seq = self.sequences[si % len(self.sequences)]
        fi  = fi % len(seq['frames'])
        key = (si, fi)
        if key in self._cache:
            return self._cache[key]

        rgb_path = os.path.join(seq['rgb_dir'], seq['frames'][fi])
        try:
            rgb = Image.open(rgb_path).convert('RGB')
        except Exception:
            return None, None

        depth = None
        if seq['depth_dir']:
            dname = seq['frames'][fi].replace('.png', '_depth.tiff')
            dp    = os.path.join(seq['depth_dir'], dname)
            if os.path.exists(dp):
                try:
                    depth = Image.open(dp)
                except Exception:
                    pass

        # limit cache size
        if len(self._cache) > 60:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (rgb, depth)
        return rgb, depth

    def next_frame(self):
        self.frame_idx = (self.frame_idx + 1) % max(1, self.n_frames)

    def prev_frame(self):
        self.frame_idx = (self.frame_idx - 1) % max(1, self.n_frames)

    def next_seq(self):
        if self.sequences:
            self.seq_idx   = (self.seq_idx + 1) % len(self.sequences)
            self.frame_idx = 0

    def prev_seq(self):
        if self.sequences:
            self.seq_idx   = (self.seq_idx - 1) % len(self.sequences)
            self.frame_idx = 0


# ── synthetic frame generator (demo mode) ────────────────────────────────────

class SyntheticGenerator:
    """Renders realistic-looking colonoscopy frames when no real data is available."""

    PALETTE = [
        ((190,100,80), (140,60,45)),
        ((210,130,100),(160,80,55)),
        ((175,90,70),  (120,50,35)),
        ((200,115,90), (150,70,50)),
    ]

    def __init__(self):
        self._t     = 0
        self._scene = random.randint(0, len(self.PALETTE)-1)
        self.frame_idx = 0
        self.n_frames  = 120

    def get_frame(self, frame_idx=None):
        fi = frame_idx if frame_idx is not None else self.frame_idx
        self._t = fi * 3
        return self._render(fi), None

    def _render(self, fi):
        W = H = 640
        t = fi * 3
        img = Image.new('RGB', (W, H), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        pal = self.PALETTE[self._scene % len(self.PALETTE)]
        light, dark = pal

        # --- colon lumen background ---
        for step in range(28, 0, -1):
            r_val = step * (W // 2) // 28
            blend = step / 28
            col = tuple(int(light[c]*blend + dark[c]*(1-blend)) for c in range(3))
            draw.ellipse([W//2-r_val, H//2-r_val, W//2+r_val, H//2+r_val], fill=col)

        # --- haustral folds (circular ridges) ---
        for i in range(5):
            angle_off = (i / 5) * math.pi * 2 + t * 0.015
            r_fold = 80 + i * 28
            fold_col = tuple(max(0, c - 35) for c in light)
            for a in range(0, 360, 4):
                rad = math.radians(a)
                warp = r_fold + 12 * math.sin(rad * 4 + i + t * 0.02)
                x = W//2 + math.cos(rad + angle_off) * warp
                y = H//2 + math.sin(rad + angle_off) * warp * 0.72
                draw.ellipse([x-3, y-3, x+3, y+3], fill=fold_col)

        # --- vessels ---
        n_vessels = 4
        for i in range(n_vessels):
            ang = (i / n_vessels) * math.pi * 2 + t * 0.008
            x0  = W//2 + math.cos(ang) * 40
            y0  = H//2 + math.sin(ang) * 30
            x1  = W//2 + math.cos(ang + 0.6) * 170 + math.sin(t * 0.03 + i) * 20
            y1  = H//2 + math.sin(ang + 0.6) * 130
            v_width = max(1, int(2 + (1 + math.sin(t * 0.07 + i)) * 1.5))
            v_col   = (80, 15, 10)
            draw.line([(x0, y0), (x1, y1)], fill=v_col, width=v_width)

        # --- mucosa texture noise ---
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 9, arr.shape).astype(np.float32)
        # only inside the lumen circle
        cy, cx = np.ogrid[:H, :W]
        lumen  = ((cx - W//2)**2 + (cy - H//2)**2) < (W//2 - 30)**2
        arr[lumen] = np.clip(arr[lumen] + noise[lumen], 0, 255)

        # --- specular highlights ---
        for _ in range(3):
            hx = W//2 + random.randint(-60, 60)
            hy = H//2 + random.randint(-50, 50)
            hr = random.randint(4, 14)
            spec = np.zeros((H, W), np.float32)
            spec_mask = ((cx - hx)**2 + (cy - hy)**2) < hr**2
            arr[spec_mask] = np.minimum(255, arr[spec_mask] * 1.0 + 160)

        img = Image.fromarray(arr.astype(np.uint8))
        return img

    def next_frame(self): self.frame_idx = (self.frame_idx + 1) % self.n_frames
    def prev_frame(self): self.frame_idx = (self.frame_idx - 1) % self.n_frames


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE PROCESSING PIPELINE                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def proc_noise_reduction(img):
    return img.filter(ImageFilter.GaussianBlur(radius=1.4))

def proc_clahe(img):
    if HAS_CV2:
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        cl  = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        lab[:,:,0] = cl.apply(lab[:,:,0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return ImageEnhance.Contrast(img).enhance(1.9)

def proc_sobel(img):
    if HAS_CV2:
        arr = np.array(img.convert('L'))
        gx  = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        gy  = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(mag).convert('RGB')
    return img.filter(ImageFilter.FIND_EDGES).convert('RGB')

def proc_depth_colormap(depth_img):
    """Convert 16-bit depth TIFF to false-colour jet map."""
    if depth_img is None:
        return None
    arr = np.array(depth_img, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return None
    norm = (arr - mn) / (mx - mn)
    if HAS_CV2:
        jet = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return Image.fromarray(cv2.cvtColor(jet, cv2.COLOR_BGR2RGB))
    # fallback: greyscale
    return Image.fromarray((norm * 255).astype(np.uint8)).convert('RGB')

def extract_features(img, depth=None):
    arr = np.array(img)
    H, W = arr.shape[:2]

    # ── color ────────────────────────────────────────────────────────────────
    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    l       = 0.299*r + 0.587*g + 0.114*b
    brightness  = round(float(l.mean()), 1)
    redness_idx = round(float(r.mean() / (g.mean()+1)), 3)

    if HAS_CV2:
        hsv     = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h_hist, bins = np.histogram(hsv[:,:,0].flatten(), 36, (0,180))
        dom_hue = int(bins[np.argmax(h_hist)] * 2)
        mean_sat = round(float(hsv[:,:,1].mean()), 1)
        mean_val = round(float(hsv[:,:,2].mean()), 1)
    else:
        dom_hue = 0; mean_sat = 0; mean_val = round(brightness, 1)

    # ── texture ──────────────────────────────────────────────────────────────
    gray = np.array(img.convert('L'))
    if HAS_SKI:
        lbp      = local_binary_pattern(gray, P=8, R=1, method='uniform')
        h_lbp, _ = np.histogram(lbp.flatten(), 10, (0,10), density=True)
        lbp_unif = round(float(np.sum(h_lbp**2)), 5)
    else:
        lbp_unif = 0.0

    shifted  = np.roll(gray, 1, axis=1).astype(float)
    glcm_con = round(float(((gray.astype(float) - shifted)**2).mean()), 2)

    # ── contrast ─────────────────────────────────────────────────────────────
    mn_l, mx_l = l.min(), l.max()
    contrast_r = round((mx_l / (mn_l+1)), 2)

    # ── edge density ─────────────────────────────────────────────────────────
    if HAS_CV2:
        edges    = cv2.Canny(gray, 40, 120)
        edge_den = round(float((edges > 0).mean() * 100), 1)
    else:
        edge_den = 0.0

    # ── depth stats ──────────────────────────────────────────────────────────
    depth_mean = depth_std = None
    if depth is not None:
        d_arr      = np.array(depth, dtype=np.float32)
        # scale: pixel 16384 → 25mm  (16-bit linear 0-100mm)
        mm         = d_arr * 100.0 / 65535.0
        depth_mean = round(float(mm.mean()), 1)
        depth_std  = round(float(mm.std()),  1)

    return {
        'brightness':  brightness,
        'contrast_r':  contrast_r,
        'edge_density':edge_den,
        'dom_hue':     dom_hue,
        'mean_sat':    mean_sat,
        'mean_val':    mean_val,
        'redness_idx': redness_idx,
        'lbp_unif':    lbp_unif,
        'glcm_con':    glcm_con,
        'depth_mean':  depth_mean,
        'depth_std':   depth_std,
    }


# ── AI Segmentation Pipeline ─────────────────────────────────────────────────

AI_MODEL = None
def load_ai_model():
    global AI_MODEL
    if AI_MODEL is not None:
        return True
    if not HAS_TORCH:
        return False
    try:
        from Model_train import UNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet().to(device)
        # Check for both standard and typo filenames 
        model_path = os.path.join(BASE, 'unet_kvasir_best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE, 'unet_ksavir_best.pth')
        if not os.path.exists(model_path):
            print(f"AI Model not found at {model_path}")
            return False
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        AI_MODEL = (model, device)
        print(f"AI Model loaded successfully on {device}.")
        return True
    except Exception as e:
        print(f"Error loading AI model: {e}")
        return False

def proc_ai_seg(img_pil):
    if not load_ai_model():
        return img_pil
    
    model, device = AI_MODEL
    img_np = np.array(img_pil.convert('RGB'))
    orig_h, orig_w = img_np.shape[:2]
    
    img_resized = cv2.resize(img_np, (256, 256)) if HAS_CV2 else np.array(img_pil.resize((256, 256), Image.BILINEAR))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    input_batch = torch.tensor(img_tensor).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        prob_mask = model(input_batch).squeeze().cpu().numpy()
        binary_mask = (prob_mask > 0.5).astype(np.uint8)
        
    mask_resized = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) if HAS_CV2 else np.array(Image.fromarray(binary_mask).resize((orig_w, orig_h), Image.NEAREST))
    
    overlay = np.zeros_like(img_np)
    overlay[mask_resized == 1] = [0, 255, 100]  # Bright green overlay
    
    blended = cv2.addWeighted(img_np, 1.0, overlay, 0.4, 0) if HAS_CV2 else np.where(mask_resized[:,:,None] == 1, (img_np * 0.6 + overlay * 0.4).astype(np.uint8), img_np)
    return Image.fromarray(blended)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCOPE VIEWPORT WIDGET                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ScopeViewport(tk.Canvas):
    """
    Main circular scope viewport.
    Composites the real frame + processing + HUD overlay.
    """
    DIAM = 500
    R    = 240

    def __init__(self, parent, app, **kw):
        super().__init__(parent, width=self.DIAM, height=self.DIAM,
                         bg='#000', highlightthickness=0, **kw)
        self.app      = app
        self._tk_img  = None
        self._flash   = 0
        self._scan_y  = 0
        self._render_pending = False

    def render(self, rgb_pil, depth_pil, state):
        """Composite and display one frame."""
        W = H = self.DIAM

        if rgb_pil is None:
            self._show_blank()
            return

        # ── resize to fill scope circle ───────────────────────────────────
        img = rgb_pil.copy()
        img = img.resize((W, H), Image.LANCZOS)

        # ── zoom + tip deflection ─────────────────────────────────────────
        zf = state['zoom']
        ud = state['ud_defl']
        rl = state['rl_defl']
        
        if zf > 1.0 or ud != 0 or rl != 0:
            # Shift the center of view to simulate bending the scope tip
            cx = W / 2 + (rl / 180.0) * (W / 2.0)
            cy = H / 2 + (ud / 180.0) * (H / 2.0)
            
            bw = W / zf
            bh = H / zf
            
            left = int(cx - bw / 2)
            top  = int(cy - bh / 2)
            right = int(cx + bw / 2)
            bottom = int(cy + bh / 2)
            
            img = img.crop((left, top, right, bottom))
            img = img.resize((W, H), Image.LANCZOS)

        # ── LED mode colour matrix ────────────────────────────────────────
        mode = state['mode']
        if mode == 'nbi':
            r, g, b = img.split()
            r = r.point(lambda p: int(p * 0.3))
            g = g.point(lambda p: int(p * 0.8))
            b = b.point(lambda p: min(255, int(p * 1.5)))
            img = Image.merge('RGB', (r, g, b))
        elif mode == 'wli+':
            r, g, b = img.split()
            r = r.point(lambda p: min(255, int(p * 1.2)))
            g = g.point(lambda p: min(255, int(p * 1.1)))
            img = Image.merge('RGB', (r, g, b))

        # ── intensity ─────────────────────────────────────────────────────
        iz  = state['intensity'] / 100.0
        img = ImageEnhance.Brightness(img).enhance(iz)

        # ── processing passes ─────────────────────────────────────────────
        proc = state['proc']
        if proc.get('nr'):
            img = proc_noise_reduction(img)
        if proc.get('clahe'):
            img = proc_clahe(img)
        if proc.get('edge'):
            img = proc_sobel(img)
        if proc.get('depth') and depth_pil is not None:
            depth_col = proc_depth_colormap(depth_pil)
            if depth_col:
                dc  = depth_col.resize((W, H), Image.LANCZOS)
                img = Image.blend(img, dc, alpha=0.55)
        if proc.get('inv'):
            img = Image.fromarray(255 - np.array(img))
        if proc.get('ai'):
            img = proc_ai_seg(img)

        # ── focus blur ────────────────────────────────────────────────────
        blur = abs(state['focus'] - 50) / 50 * 3.5
        if blur > 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))

        # ── circular crop ─────────────────────────────────────────────────
        mask_c = Image.new('L', (W, H), 0)
        ImageDraw.Draw(mask_c).ellipse([W//2 - self.R, H//2 - self.R,
                                        W//2 + self.R, H//2 + self.R], fill=255)
        black  = Image.new('RGB', (W, H), 0)
        img    = Image.composite(img, black, mask_c)

        # ── vignette ─────────────────────────────────────────────────────
        vign = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        vd   = ImageDraw.Draw(vign)
        for s in range(35, 0, -1):
            a  = int((35 - s) * 5.8)
            rr = self.R - s * 4
            if rr > 0:
                vd.ellipse([W//2 - rr, H//2 - rr, W//2 + rr, H//2 + rr],
                           outline=(0, 0, 0, a), width=8)
        img_rgba = img.convert('RGBA')
        img_rgba.alpha_composite(vign)
        img = img_rgba.convert('RGB')

        # ── scanline ─────────────────────────────────────────────────────
        self._scan_y = (self._scan_y + 2) % H
        scan = ImageDraw.Draw(img)
        scan.line([(0, self._scan_y), (W, self._scan_y)],
                  fill=(0, 232, 122, 12), width=1)

        # ── capture flash ─────────────────────────────────────────────────
        if self._flash > 0:
            overlay = Image.new('RGB', (W, H), (255, 255, 255))
            img = Image.blend(img, overlay, alpha=self._flash / 10)
            self._flash = max(0, self._flash - 1)

        # ── HUD ───────────────────────────────────────────────────────────
        hud = ImageDraw.Draw(img)
        seq_name = state.get('seq_name', 'DEMO')
        fi       = state.get('frame_idx', 0)
        n_fr     = state.get('n_frames', 0)
        pose_str = state.get('pose_str', '—')

        hud_l = [
            f"U/D    {state['ud_defl']:>4}°",
            f"R/L    {state['rl_defl']:>4}°",
            f"ZOOM   {state['zoom']:.1f}×",
            f"FOCUS  {state['focus']}",
        ]
        hud_r = [
            f"MODE  {mode.upper()}",
            f"INT   {state['intensity']}%",
            f"{state['ct']}K",
            f"FPS   30",
        ]
        hud_b = [
            f"SEQ: {seq_name[:22]}",
            f"FRAME: {fi+1:04d}/{n_fr:04d}",
        ]
        if pose_str != '—':
            hud_b.append(f"POSE: {pose_str}")

        green = (0, 232, 122)
        dim   = (0, 120, 60)
        for i, line in enumerate(hud_l):
            hud.text((14, 14 + i*17), line, fill=green)
        for i, line in enumerate(hud_r):
            x = W - 14 - len(line) * 6
            hud.text((x, 14 + i*17), line, fill=green)
        for i, line in enumerate(hud_b):
            hud.text((14, H - 18 - (len(hud_b)-1-i)*15), line, fill=dim)

        # crosshair
        cx, cy = W // 2, H // 2
        for dx, dy in [(-20,0),( 20,0),(0,-20),(0, 20)]:
            hud.line([(cx, cy), (cx+dx, cy+dy)], fill=(0, 232, 122, 120), width=1)
        hud.ellipse([cx-22, cy-22, cx+22, cy+22], outline=(0, 232, 122, 80), width=1)

        self._tk_img = ImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor='nw', image=self._tk_img)

    def _show_blank(self):
        W = H = self.DIAM
        img = Image.new('RGB', (W, H), 0)
        d   = ImageDraw.Draw(img)
        d.ellipse([W//2 - self.R, H//2 - self.R, W//2 + self.R, H//2 + self.R],
                  outline=(30, 60, 80), width=2)
        d.text((W//2 - 80, H//2 - 10), "No frame available", fill=(40, 80, 100))
        self._tk_img = ImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor='nw', image=self._tk_img)

    def flash(self):
        self._flash = 10


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FILM STRIP                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class FilmStrip(tk.Frame):
    """Horizontal thumbnail strip of frames from the current sequence."""
    THUMB = 64
    MAX_THUMBS = 20

    def __init__(self, parent, app):
        super().__init__(parent, bg='#040608', height=self.THUMB + 12)
        self.app      = app
        self._thumbs  = []
        self._btns    = []
        self._tk_imgs = []
        self.pack_propagate(False)

        inner = tk.Frame(self, bg='#040608')
        inner.pack(fill='both', expand=True)

        sb = tk.Scrollbar(inner, orient='horizontal', bg=PANEL)
        sb.pack(side='bottom', fill='x')
        self._canvas = tk.Canvas(inner, bg='#040608', highlightthickness=0,
                                 height=self.THUMB + 8, xscrollcommand=sb.set)
        self._canvas.pack(fill='both', expand=True)
        sb.config(command=self._canvas.xview)

        self._inner = tk.Frame(self._canvas, bg='#040608')
        self._canvas.create_window((0, 0), window=self._inner, anchor='nw')
        self._inner.bind('<Configure>',
                         lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))

    def rebuild(self, dataset, n_show=None):
        for w in self._inner.winfo_children():
            w.destroy()
        self._btns = []; self._tk_imgs = []

        seq = dataset.current_seq
        if seq is None:
            return

        frames = seq['frames']
        total  = len(frames)
        step   = max(1, total // self.MAX_THUMBS)
        idxs   = list(range(0, total, step))[:self.MAX_THUMBS]

        for fi in idxs:
            rgb, _ = dataset.get_frame(frame_idx=fi)
            if rgb is None:
                continue
            thumb = rgb.copy()
            # circular crop for thumb
            thumb.thumbnail((self.THUMB, self.THUMB), Image.LANCZOS)
            tw, th = thumb.size
            mask_t = Image.new('L', (tw, th), 0)
            ImageDraw.Draw(mask_t).ellipse([0, 0, tw, th], fill=255)
            black_t = Image.new('RGB', (tw, th), 0)
            thumb   = Image.composite(thumb, black_t, mask_t)

            tk_img = ImageTk.PhotoImage(thumb)
            self._tk_imgs.append(tk_img)

            is_active = (fi == dataset.frame_idx)
            border_c  = '#4d9de0' if is_active else '#1c2535'

            btn = tk.Label(self._inner, image=tk_img, bg=border_c,
                           cursor='hand2', relief='flat', bd=2)
            btn.pack(side='left', padx=3, pady=4)
            btn.bind('<Button-1>', lambda e, f=fi: self.app.jump_to_frame(f))
            self._btns.append((fi, btn))

    def highlight(self, frame_idx):
        for fi, btn in self._btns:
            btn.config(bg='#4d9de0' if fi == frame_idx else '#1c2535')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN APPLICATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class EndoscopeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("C3VD Intelligent Endoscope Simulator  —  SBE3220")
        self.configure(bg=BG)
        self.resizable(True, True)

        # dataset
        self.dataset = C3VDDataset()
        self.synth   = SyntheticGenerator()
        self.demo    = not self.dataset.has_real_data

        if self.demo:
            print(MANUAL_GUIDE)

        # scope state
        self.state = {
            'intensity': 80, 'ct': 5600, 'gamma': 1.0,
            'mode': 'white', 'zoom': 1.0, 'focus': 50,
            'depth': 0, 'ud_defl': 0, 'rl_defl': 0,
            'proc': {k: False for k in ['nr','clahe','edge','depth','inv','ai']},
            'seq_name': 'DEMO' if self.demo else '',
            'frame_idx': 0, 'n_frames': 0, 'pose_str': '—',
        }
        self._nav   = {k: False for k in ['u','d','l','r']}
        self._auto  = False
        self._auto_id = None
        self._frame_features = {}
        self._captured = 0
        self._last_feat_t = 0

        self._build_ui()
        self._bind_keys()
        self._refresh_sequence()
        self._loop()

    # ── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.geometry('1320x820')

        # header bar
        hdr = tk.Frame(self, bg='#03050a', height=44)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='◉  C3VD ENDOSCOPE SIMULATOR  —  SBE3220',
                 bg='#03050a', fg=ACC, font=FNT_H).pack(side='left', padx=14, pady=10)
        mode_lbl = '  DEMO MODE — no C3VD data found' if self.demo else f'  {len(self.dataset.sequences)} sequences loaded'
        mode_col = ACC3 if self.demo else ACC2
        self._header_status = tk.Label(hdr, text=mode_lbl, bg='#03050a',
                                       fg=mode_col, font=FNT_S)
        self._header_status.pack(side='left', padx=4)
        tk.Label(hdr, text='C3VD  ·  Johns Hopkins University  ·  CC BY-NC-SA 4.0',
                 bg='#03050a', fg=TXT2, font=FNT_S).pack(side='right', padx=14)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=TXT, padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', BORDER)], foreground=[('selected', ACC)])

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)

        body = tk.Frame(notebook, bg=BG)
        notebook.add(body, text="Simulator")

        pred_tab = tk.Frame(notebook, bg=BG)
        notebook.add(pred_tab, text="Single Image Prediction")
        self._build_pred_tab(pred_tab)

        # left panel
        self._left = tk.Frame(body, bg=PANEL, width=220)
        self._left.pack(side='left', fill='y', padx=(6,3), pady=6)
        self._left.pack_propagate(False)
        self._build_left()

        # centre
        centre = tk.Frame(body, bg=BG)
        centre.pack(side='left', fill='both', expand=True, pady=6)

        # sequence selector bar
        seq_bar = tk.Frame(centre, bg=PANEL, height=38)
        seq_bar.pack(fill='x', pady=(0,3))
        seq_bar.pack_propagate(False)
        tk.Label(seq_bar, text='SEQ:', bg=PANEL, fg=TXT2, font=FNT_S).pack(side='left', padx=8, pady=8)
        self._seq_label = tk.Label(seq_bar, text='—', bg=PANEL, fg=TXT, font=FNT_S)
        self._seq_label.pack(side='left')
        tk.Button(seq_bar, text='◀ Prev Seq', font=FNT_S, bg=CARD, fg=TXT,
                  relief='flat', bd=0, padx=8, pady=4,
                  command=self.prev_seq).pack(side='right', padx=6, pady=4)
        tk.Button(seq_bar, text='Next Seq ▶', font=FNT_S, bg=CARD, fg=TXT,
                  relief='flat', bd=0, padx=8, pady=4,
                  command=self.next_seq).pack(side='right', padx=2, pady=4)
        if self.demo:
            tk.Button(seq_bar, text='Install Dataset', font=FNT_S, bg='#1a3020', fg=ACC2,
                      relief='flat', bd=0, padx=10, pady=4,
                      command=self._show_install_guide).pack(side='right', padx=8, pady=4)

        # scope viewport
        scope_frame = tk.Frame(centre, bg=BG)
        scope_frame.pack(fill='both', expand=True)
        self.viewport = ScopeViewport(scope_frame, self)
        self.viewport.pack(pady=4)

        # toolbar
        tb = tk.Frame(centre, bg=PANEL, height=44)
        tb.pack(fill='x', pady=(2,0))
        tb.pack_propagate(False)
        tk.Label(tb, text='PROCESS:', bg=PANEL, fg=TXT2, font=FNT_S).pack(side='left', padx=(10,4), pady=10)
        self._proc_btns = {}
        for key, lbl in [('nr','Noise↓'),('clahe','CLAHE'),('edge','Edges'),
                         ('depth','Depth'),('inv','Invert'),('ai','AI Seg')]:
            b = tk.Button(tb, text=lbl, font=FNT_S, bg=CARD, fg=TXT2,
                          relief='flat', bd=0, padx=8, pady=5,
                          command=lambda k=key: self._toggle_proc(k))
            b.pack(side='left', padx=3, pady=7)
            self._proc_btns[key] = b

        tk.Frame(tb, bg=BORDER, width=1).pack(side='left', fill='y', padx=6, pady=6)
        tk.Button(tb, text='⬤  Capture', font=FNT_S, bg='#0a2018', fg=ACC2,
                  relief='flat', bd=0, padx=10, pady=5,
                  command=self.capture).pack(side='left', padx=4, pady=7)
        self._auto_btn = tk.Button(tb, text='Auto ▶', font=FNT_S, bg=CARD, fg=TXT2,
                                   relief='flat', bd=0, padx=8, pady=5,
                                   command=self._toggle_auto)
        self._auto_btn.pack(side='left', padx=3, pady=7)

        tk.Button(tb, text='◀ Frame', font=FNT_S, bg=CARD, fg=TXT2,
                  relief='flat', bd=0, padx=8, pady=5,
                  command=self.prev_frame).pack(side='right', padx=3, pady=7)
        tk.Button(tb, text='Frame ▶', font=FNT_S, bg=CARD, fg=TXT2,
                  relief='flat', bd=0, padx=8, pady=5,
                  command=self.next_frame).pack(side='right', padx=3, pady=7)

        # film strip
        self.strip = FilmStrip(centre, self)
        self.strip.pack(fill='x', pady=(3,0))

        # right panel
        self._right = tk.Frame(body, bg=PANEL, width=210)
        self._right.pack(side='right', fill='y', padx=(3,6), pady=6)
        self._right.pack_propagate(False)
        self._build_right()

        # status bar
        sb = tk.Frame(self, bg='#03050a', height=22)
        sb.pack(fill='x', side='bottom')
        self._status = tk.Label(sb, text='Ready', bg='#03050a', fg=TXT2, font=FNT_S)
        self._status.pack(side='left', padx=10, pady=3)

    def _build_left(self):
        def sec(t):
            tk.Frame(self._left, bg=BORDER, height=1).pack(fill='x', padx=8, pady=(10,2))
            tk.Label(self._left, text=t, bg=PANEL, fg=TXT2, font=FNT_S).pack(anchor='w', padx=10)

        def slider(label, key, mn, mx, val, fmt):
            tk.Label(self._left, text=label, bg=PANEL, fg=TXT2, font=FNT_S).pack(anchor='w', padx=10, pady=(5,0))
            var = tk.IntVar(value=val)
            lbl = tk.Label(self._left, text=fmt(val), bg=PANEL, fg=ACC, font=FNT_S)
            lbl.pack(anchor='e', padx=10)
            sl  = ttk.Scale(self._left, from_=mn, to=mx, variable=var, orient='horizontal')
            sl.pack(fill='x', padx=10, pady=(0,2))
            def on_change(*_):
                v = int(var.get())
                self.state[key] = v
                lbl.config(text=fmt(v))
            var.trace_add('write', on_change)
            return var

        sec('ILLUMINATION')
        self._sv_int   = slider('Intensity',    'intensity', 0,   100,  80,  lambda v:f'{v}%')
        self._sv_ct    = slider('Color temp',   'ct',        3200,7500, 5600,lambda v:f'{v}K')
        self._sv_focus = slider('Focus',        'focus',     0,   100,  50,  lambda v:str(v))

        sec('LED MODE')
        mf = tk.Frame(self._left, bg=PANEL); mf.pack(fill='x', padx=8, pady=5)
        self._mode_btns = {}
        for m in ['white','nbi','wli+']:
            b = tk.Button(mf, text=m.upper(), font=FNT_S, bg=CARD, fg=TXT2,
                          relief='flat', bd=0, padx=6, pady=4,
                          command=lambda x=m: self._set_mode(x))
            b.pack(side='left', padx=2)
            self._mode_btns[m] = b
        self._set_mode('white')

        sec('DEFLECTION KNOBS')
        dpad = tk.Frame(self._left, bg=PANEL); dpad.pack(pady=6)
        dirs = [('',0,0),('U',0,1,'u'),('',0,2),
                ('L',1,0,'l'),('·',1,1,'c'),('R',1,2,'r'),
                ('',2,0),('D',2,1,'d'),('',2,2)]
        for item in dirs:
            if len(item) == 2 or not item[0]: continue
            lbl_, r, c = item[0], item[1], item[2]
            d_key = item[3] if len(item) > 3 else None
            b = tk.Button(dpad, text=lbl_, width=3, height=1,
                          bg=CARD, fg=TXT, font=('Consolas', 12), relief='flat', bd=0)
            if d_key:
                if d_key == 'c':
                    b.bind('<ButtonPress-1>', lambda e: self._center_defl())
                else:
                    b.bind('<ButtonPress-1>',   lambda e, k=d_key: self._nav_set(k, True))
                    b.bind('<ButtonRelease-1>', lambda e, k=d_key: self._nav_set(k, False))
            b.grid(row=r, column=c, padx=2, pady=2, in_=dpad)
            
        self._sv_ud_defl = slider('U/D Angle', 'ud_defl', -180, 180, 0, lambda v:f'{v}°')
        self._sv_rl_defl = slider('R/L Angle', 'rl_defl', -180, 180, 0, lambda v:f'{v}°')

        sec('IMAGING')
        self._sv_zoom  = slider('Zoom', '_zoom_int', 10, 35, 10, lambda v:f'{v/10:.1f}×')
        self.state['_zoom_int'] = 10

        # watch _zoom_int to update zoom
        def _zoom_watch(*_):
            self.state['zoom'] = self.state['_zoom_int'] / 10
        self._sv_zoom.trace_add('write', _zoom_watch)

    def _center_defl(self):
        self._sv_ud_defl.set(0)
        self._sv_rl_defl.set(0)

    def _build_right(self):
        def sec(t):
            tk.Frame(self._right, bg=BORDER, height=1).pack(fill='x', padx=8, pady=(10,2))
            tk.Label(self._right, text=t, bg=PANEL, fg=TXT2, font=FNT_S).pack(anchor='w', padx=10)

        def metric(label, color=ACC):
            f = tk.Frame(self._right, bg=CARD)
            f.pack(fill='x', padx=8, pady=3)
            tk.Label(f, text=label, bg=CARD, fg=TXT2, font=FNT_S).pack(anchor='w', padx=6, pady=(4,0))
            lbl = tk.Label(f, text='—', bg=CARD, fg=color, font=FNT_L)
            lbl.pack(anchor='w', padx=6, pady=(0,4))
            return lbl

        def feat(label):
            f = tk.Frame(self._right, bg=CARD)
            f.pack(fill='x', padx=8, pady=2)
            tk.Label(f, text=label, bg=CARD, fg=TXT2, font=FNT_S, anchor='w').pack(side='left', padx=6, pady=4)
            lbl = tk.Label(f, text='—', bg=CARD, fg=ACC, font=FNT_S)
            lbl.pack(side='right', padx=6)
            return lbl

        sec('LIVE METRICS')
        self._m_bright   = metric('Brightness',   ACC2)
        self._m_contrast = metric('Contrast',     ACC)
        self._m_edge     = metric('Edge density', '#e3b341')
        self._m_depth    = metric('Depth (mm)',   '#a78bfa')

        sec('SHAPE FEATURES')
        self._f_area   = feat('Area (px)')
        self._f_circ   = feat('Circularity')

        sec('COLOR FEATURES')
        self._f_hue    = feat('Dom. hue')
        self._f_sat    = feat('Mean sat')
        self._f_red    = feat('Redness idx')

        sec('TEXTURE FEATURES')
        self._f_lbp    = feat('LBP uniformity')
        self._f_glcm   = feat('GLCM contrast')

        sec('POSE (camera→world)')
        self._pose_lbl = tk.Label(self._right, text='—', bg=PANEL, fg=TXT2,
                                  font=('Consolas', 8), justify='left')
        self._pose_lbl.pack(anchor='w', padx=10, pady=4)

        sec('EVENT LOG')
        self._log = tk.Text(self._right, height=8, bg=CARD, fg=ACC2, font=FNT_S,
                            relief='flat', bd=0, state='disabled', wrap='word')
        self._log.pack(fill='x', padx=8, pady=4)
        self._log_msg('System started')
        mode_txt = 'DEMO mode — synthetic frames' if self.demo else f'{len(self.dataset.sequences)} C3VD sequences'
        self._log_msg(mode_txt)
        if self.demo:
            self._log_msg('Run: python install_dataset.py')

    # ── PREDICTION TAB ───────────────────────────────────────────────────────

    def _build_pred_tab(self, parent):
        top_bar = tk.Frame(parent, bg=PANEL, height=50)
        top_bar.pack(fill='x', side='top')
        top_bar.pack_propagate(False)

        tk.Button(top_bar, text="Load Image", font=FNT_S, bg=CARD, fg=TXT,
                  relief='flat', bd=0, padx=10, pady=5,
                  command=self._load_pred_image).pack(side='left', padx=10, pady=10)
        tk.Button(top_bar, text="Predict", font=FNT_S, bg='#0a2018', fg=HUD_G,
                  relief='flat', bd=0, padx=10, pady=5,
                  command=self._run_prediction).pack(side='left', padx=10, pady=10)

        self._pred_status = tk.Label(top_bar, text="Load an image to start.", bg=PANEL, fg=TXT2, font=FNT_S)
        self._pred_status.pack(side='left', padx=10, pady=10)

        content = tk.Frame(parent, bg=BG)
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        self._pred_img_lbl = tk.Label(content, bg=BG)
        self._pred_img_lbl.pack(side='left', expand=True)
        
        self._pred_res_lbl = tk.Label(content, bg=BG)
        self._pred_res_lbl.pack(side='right', expand=True)
        
        self._pred_image = None
        self._pred_tk_img = None
        self._pred_tk_res = None

    def _load_pred_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if path:
            try:
                self._pred_image = Image.open(path).convert('RGB')
                self._show_pred_images(self._pred_image, None)
                self._pred_status.config(text=f"Loaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _show_pred_images(self, orig, res):
        def _to_tk(img):
            if img is None: return None
            w, h = img.size
            ratio = min(500/w, 500/h) if w > 0 and h > 0 else 1
            nw, nh = max(1, int(w*ratio)), max(1, int(h*ratio))
            img = img.resize((nw, nh), Image.LANCZOS)
            return ImageTk.PhotoImage(img)

        self._pred_tk_img = _to_tk(orig)
        self._pred_img_lbl.config(image=self._pred_tk_img)

        if res is not None:
            self._pred_tk_res = _to_tk(res)
            self._pred_res_lbl.config(image=self._pred_tk_res)
        else:
            self._pred_res_lbl.config(image='')

    def _run_prediction(self):
        if self._pred_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
            
        self._pred_status.config(text="Running prediction...")
        self.update()
        
        res = proc_ai_seg(self._pred_image)
        self._show_pred_images(self._pred_image, res)
        self._pred_status.config(text="Prediction complete.")

    # ── KEY BINDINGS ─────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.bind_all('<KeyPress-Up>',    lambda e: self._nav_set('u', True))
        self.bind_all('<KeyRelease-Up>',  lambda e: self._nav_set('u', False))
        self.bind_all('<KeyPress-Down>',  lambda e: self._nav_set('d', True))
        self.bind_all('<KeyRelease-Down>',lambda e: self._nav_set('d', False))
        self.bind_all('<KeyPress-Left>',  lambda e: self._nav_set('l', True))
        self.bind_all('<KeyRelease-Left>',lambda e: self._nav_set('l', False))
        self.bind_all('<KeyPress-Right>', lambda e: self._nav_set('r', True))
        self.bind_all('<KeyRelease-Right>',lambda e: self._nav_set('r', False))
        self.bind_all('<comma>',          lambda e: self.prev_frame())
        self.bind_all('<period>',         lambda e: self.next_frame())
        self.bind_all('<a>',              lambda e: self.prev_frame())
        self.bind_all('<d>',              lambda e: self.next_frame())
        self.bind_all('<a>',              lambda e: self.prev_seq())
        self.bind_all('<d>',              lambda e: self.next_seq())
        self.bind_all('<A>',              lambda e: self.prev_seq())
        self.bind_all('<D>',              lambda e: self.next_seq())
        self.bind_all('<bracketleft>',    lambda e: self.prev_seq())
        self.bind_all('<bracketright>',   lambda e: self.next_seq())
        self.bind_all('<space>',          lambda e: self.capture())
        self.bind_all('<Escape>',         lambda e: self.destroy())
        self.bind_all('<plus>',           lambda e: self._zoom_step(1))
        self.bind_all('<minus>',          lambda e: self._zoom_step(-1))
        # number keys toggle proc
        for i, k in enumerate(['nr','clahe','edge','depth','inv','ai'], 1):
            self.bind_all(str(i), lambda e, x=k: self._toggle_proc(x))

    def _nav_set(self, key, val):
        self._nav[key] = val

    def _zoom_step(self, d):
        v = max(10, min(35, self.state['_zoom_int'] + d * 2))
        self.state['_zoom_int'] = v
        self.state['zoom']      = v / 10
        self._sv_zoom.set(v)

    # ── STATE UPDATES ────────────────────────────────────────────────────────

    def _set_mode(self, m):
        self.state['mode'] = m
        for k, b in self._mode_btns.items():
            b.config(bg='#0c2a20' if k==m else CARD,
                     fg=ACC2     if k==m else TXT2)

    def _toggle_proc(self, key):
        self.state['proc'][key] = not self.state['proc'][key]
        on = self.state['proc'][key]
        self._proc_btns[key].config(bg='#0c2030' if on else CARD,
                                    fg=ACC       if on else TXT2)
        self._log_msg(f"{key.upper()}: {'ON' if on else 'OFF'}")

    def _toggle_auto(self):
        self._auto = not self._auto
        self._auto_btn.config(
            bg='#0c2030' if self._auto else CARD,
            fg=ACC       if self._auto else TXT2,
            text='Stop ■' if self._auto else 'Auto ▶')
        self._log_msg(f"Auto-advance: {'ON' if self._auto else 'OFF'}")

    def _log_msg(self, msg):
        self._log.config(state='normal')
        ts = time.strftime('%H:%M:%S')
        self._log.insert('end', f"{ts}  {msg}\n")
        self._log.see('end')
        self._log.config(state='disabled')
        if int(self._log.index('end').split('.')[0]) > 80:
            self._log.config(state='normal')
            self._log.delete('1.0', '2.0')
            self._log.config(state='disabled')

    def _set_status(self, msg):
        self._status.config(text=msg)

    # ── SEQUENCE / FRAME NAVIGATION ──────────────────────────────────────────

    def _refresh_sequence(self):
        if self.demo:
            self.state['seq_name'] = 'DEMO (synthetic)'
            self.state['n_frames'] = self.synth.n_frames
            self._seq_label.config(text='DEMO MODE — synthetic colonoscopy frames')
        else:
            seq = self.dataset.current_seq
            if seq:
                self.state['seq_name'] = seq['name']
                self.state['n_frames'] = len(seq['frames'])
                self._seq_label.config(text=f"{seq['name']}  ({len(seq['frames'])} frames)")
            self.strip.rebuild(self.dataset)
        self._log_msg(f"Seq: {self.state['seq_name']}")

    def next_frame(self):
        if self.demo:
            self.synth.next_frame()
            self.state['frame_idx'] = self.synth.frame_idx
        else:
            self.dataset.next_frame()
            self.state['frame_idx'] = self.dataset.frame_idx
            self.strip.highlight(self.dataset.frame_idx)

    def prev_frame(self):
        if self.demo:
            self.synth.prev_frame()
            self.state['frame_idx'] = self.synth.frame_idx
        else:
            self.dataset.prev_frame()
            self.state['frame_idx'] = self.dataset.frame_idx
            self.strip.highlight(self.dataset.frame_idx)

    def jump_to_frame(self, fi):
        if not self.demo:
            self.dataset.frame_idx = fi
            self.state['frame_idx'] = fi
            self.strip.highlight(fi)

    def next_seq(self):
        if not self.demo:
            self.dataset.next_seq()
            self._refresh_sequence()

    def prev_seq(self):
        if not self.demo:
            self.dataset.prev_seq()
            self._refresh_sequence()

    def capture(self):
        self._captured += 1
        self.viewport.flash()
        self._log_msg(f"Captured #{self._captured}  frame {self.state['frame_idx']+1:04d}")

    # ── MAIN LOOP ────────────────────────────────────────────────────────────

    def _loop(self):
        # pan from nav (tip deflection knobs)
        spd = 3
        if self._nav['u']: self._sv_ud_defl.set(max(-180, self.state['ud_defl'] - spd))
        if self._nav['d']: self._sv_ud_defl.set(min( 180, self.state['ud_defl'] + spd))
        if self._nav['l']: self._sv_rl_defl.set(max(-180, self.state['rl_defl'] - spd))
        if self._nav['r']: self._sv_rl_defl.set(min( 180, self.state['rl_defl'] + spd))

        # auto-advance
        if self._auto:
            if not hasattr(self, '_auto_tick'):
                self._auto_tick = 0
            self._auto_tick += 1
            if self._auto_tick >= 8:   # ~every 8 * 33ms ≈ 4fps advance
                self._auto_tick = 0
                self.next_frame()

        # get frame
        if self.demo:
            rgb, depth = self.synth.get_frame(self.synth.frame_idx)
        else:
            rgb, depth = self.dataset.get_frame()
            # update pose display
            pose = self.dataset.current_pose
            if pose is not None:
                tx, ty, tz = pose[0,3], pose[1,3], pose[2,3]
                self.state['pose_str'] = f"t=[{tx:.1f},{ty:.1f},{tz:.1f}]"
                self._pose_lbl.config(
                    text=f"tx={tx:7.3f}\nty={ty:7.3f}\ntz={tz:7.3f}")
            else:
                self.state['pose_str'] = '—'
                self._pose_lbl.config(text='—')

        # render
        self.viewport.render(rgb, depth, self.state)

        # feature extraction (throttled)
        now = time.time()
        if now - self._last_feat_t > 0.45 and rgb is not None:
            self._last_feat_t = now
            self._update_features(rgb, depth)

        self.after(33, self._loop)   # ~30 fps

    def _update_features(self, rgb, depth):
        try:
            f = extract_features(rgb, depth)
        except Exception:
            return

        self._m_bright.config(text=str(f['brightness']))
        self._m_contrast.config(text=str(f['contrast_r']))
        self._m_edge.config(text=f"{f['edge_density']}%")
        dm = f['depth_mean']
        self._m_depth.config(text=f"{dm} mm" if dm is not None else '—')

        # shape — approximate from image centre blob
        W, H = rgb.size
        self._f_area.config(text=f"{W*H//4:,}")
        self._f_circ.config(text='0.823')   # stable for scope circle

        self._f_hue.config(text=f"{f['dom_hue']}°")
        self._f_sat.config(text=str(f['mean_sat']))
        self._f_red.config(text=str(f['redness_idx']))

        self._f_lbp.config(text=str(f['lbp_unif']))
        self._f_glcm.config(text=str(f['glcm_con']))

        self._set_status(
            f"Frame {self.state['frame_idx']+1}/{self.state['n_frames']}  |  "
            f"bright={f['brightness']}  contrast={f['contrast_r']}  "
            f"edge={f['edge_density']}%")

    def _show_install_guide(self):
        win = tk.Toplevel(self)
        win.title('Dataset Installation Guide')
        win.configure(bg=BG)
        win.geometry('780x560')
        txt = tk.Text(win, bg=CARD, fg=TXT, font=FNT_S, relief='flat', bd=0, wrap='word')
        txt.pack(fill='both', expand=True, padx=12, pady=12)
        txt.insert('1.0', MANUAL_GUIDE)
        txt.config(state='disabled')
        tk.Button(win, text='Close', font=FNT_S, bg=CARD, fg=TXT, relief='flat',
                  command=win.destroy).pack(pady=8)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = EndoscopeApp()
    app.mainloop()