import os
import cv2
import numpy as np
from copy import deepcopy

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Limits for scaling and shearing to avoid extremely small/large images
MIN_SIZE = 64
MAX_SIZE = 1600


def cv2_to_pil(img):

    if img is None:
        return None
    if len(img.shape) == 2:
        return Image.fromarray(img)
    elif len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    else:
        raise ValueError("Unsupported image format")


def auto_contrast_stretch(gray):

    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    rmin = float(gray.min())
    rmax = float(gray.max())
    if rmax - rmin < 1e-6:
        return gray.copy()

    stretched = (gray - rmin) * (255.0 / (rmax - rmin))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def gamma_correction(img, gamma):

    img_f = img.astype(np.float32) / 255.0
    corrected = np.power(img_f, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return corrected


class DIPApp(tb.Window):
    def __init__(self):
        super().__init__(title="DIP Midterm - Image Processing Application",
                         themename="morph")

        self.geometry("1280x720")
        self.resizable(True, True)

        self.original_img = None
        self.current_img = None

        self.orig_photo = None
        self.proc_photo = None

        self.status_var = tk.StringVar(value="Welcome! Load an image to get started.")

        self._build_ui()

    # ---------- Status helper ----------
    def set_status(self, text: str):
        """Update the status bar text."""
        self.status_var.set(text)

    # ---------- Tab navigation helper ----------
    def show_tab(self, name: str):

        # Show/hide inner frames
        for tab_name, frame in self.tab_frames.items():
            if tab_name == name:
                frame.lift()
            else:
                frame.lower()

        # Update styles of navigation buttons
        for tab_name, btn in self.nav_buttons.items():
            if tab_name == name:
                btn.configure(bootstyle="info")               # active
            else:
                btn.configure(bootstyle="secondary-outline")  # inactive

    # ---------- UI ----------
    def _build_ui(self):

        # Top frame
        top_frame = tb.Frame(self, bootstyle=SECONDARY)
        top_frame.pack(side=TOP, fill=X)

        title_label = tb.Label(
            top_frame,
            text="Digital Image Processing Toolkit",
            font=("Segoe UI", 16, "bold"),
            anchor=W
        )
        title_label.pack(side=LEFT, padx=15, pady=10)

        btn_frame = tb.Frame(top_frame, bootstyle=SECONDARY)
        btn_frame.pack(side=RIGHT, padx=10)

        tb.Button(btn_frame, text="Open", width=10,
                  bootstyle=INFO, command=self.open_image).pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Save", width=10,
                  bootstyle=SUCCESS, command=self.save_result).pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="Reset", width=10,
                  bootstyle=WARNING, command=self.reset_image).pack(side=LEFT, padx=5)

        # Main frame
        main_frame = tb.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # preview frame
        preview_frame = tb.Labelframe(main_frame, text="Preview", bootstyle=PRIMARY)
        preview_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        self.left_panel = tb.Label(
            preview_frame,
            text="Original",
            anchor=CENTER,
            bootstyle=INVERSE
        )
        self.left_panel.pack(fill=BOTH, expand=True, padx=5, pady=5)

        self.right_panel = tb.Label(
            preview_frame,
            text="Result",
            anchor=CENTER,
            bootstyle=INVERSE
        )
        self.right_panel.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Controls frame
        control_frame = tb.Labelframe(main_frame, text="Controls", bootstyle=INFO)
        control_frame.pack(side=RIGHT, fill=Y)

        # --- Navigation bar for tabs ---
        nav_frame = tb.Frame(control_frame, bootstyle="dark")
        nav_frame.pack(fill=X, padx=5, pady=(5, 0))

        self.nav_buttons = {}
        tab_names = ["Basics", "Affine", "Intensity", "Filters", "Hist & Morph"]

        for i, name in enumerate(tab_names):
            btn = tb.Button(
                nav_frame,
                text=name,
                width=12,
                bootstyle="secondary-outline",
                command=lambda n=name: self.show_tab(n)
            )
            btn.grid(row=0, column=i, padx=2, pady=2)
            self.nav_buttons[name] = btn

        for i in range(len(tab_names)):
            nav_frame.grid_columnconfigure(i, weight=1)

        # --- Container for tab content ---
        self.tab_container = tb.Frame(control_frame, bootstyle="secondary")
        self.tab_container.pack(fill=BOTH, expand=True, padx=5, pady=5)

        self.tab_frames = {}
        for name in tab_names:
            frame = tb.Frame(self.tab_container)
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.tab_frames[name] = frame

        # ===== Basics tab =====
        basics_tab = self.tab_frames["Basics"]

        tb.Button(basics_tab, text="To Grayscale", width=18,
                  bootstyle=OUTLINE, command=self.op_to_grayscale).grid(row=0, column=0, padx=8, pady=8)
        tb.Button(basics_tab, text="Flip Horizontal", width=18,
                  bootstyle=OUTLINE, command=self.op_flip_horizontal).grid(row=0, column=1, padx=8, pady=8)
        tb.Button(basics_tab, text="Flip Vertical", width=18,
                  bootstyle=OUTLINE, command=self.op_flip_vertical).grid(row=1, column=0, padx=8, pady=8)
        tb.Button(basics_tab, text="Rotate 90 CW", width=18,
                  bootstyle=OUTLINE, command=lambda: self.op_rotate_90(True)).grid(row=1, column=1, padx=8, pady=8)
        tb.Button(basics_tab, text="Rotate 90 CCW", width=18,
                  bootstyle=OUTLINE, command=lambda: self.op_rotate_90(False)).grid(row=2, column=0, padx=8, pady=8)

        basics_tab.grid_columnconfigure(0, weight=1)
        basics_tab.grid_columnconfigure(1, weight=1)

        # ===== Affine tab =====
        affine_tab = self.tab_frames["Affine"]

        tb.Button(affine_tab, text="Rotate", width=18,
                  bootstyle=INFO, command=self.op_affine_rotate_dialog).grid(row=0, column=0, padx=8, pady=8)
        tb.Button(affine_tab, text="Scale", width=18,
                  bootstyle=INFO, command=self.op_affine_scale_dialog).grid(row=0, column=1, padx=8, pady=8)
        tb.Button(affine_tab, text="Translate", width=18,
                  bootstyle=INFO, command=self.op_affine_translate_dialog).grid(row=1, column=0, padx=8, pady=8)
        tb.Button(affine_tab, text="Shear X", width=18,
                  bootstyle=INFO, command=self.op_affine_shear_x_dialog).grid(row=1, column=1, padx=8, pady=8)
        tb.Button(affine_tab, text="Shear Y", width=18,
                  bootstyle=INFO, command=self.op_affine_shear_y_dialog).grid(row=2, column=0, padx=8, pady=8)

        affine_tab.grid_columnconfigure(0, weight=1)
        affine_tab.grid_columnconfigure(1, weight=1)

        # ===== Intensity tab =====
        intensity_tab = self.tab_frames["Intensity"]

        tb.Button(intensity_tab, text="Negative", width=18,
                  bootstyle=SECONDARY, command=self.op_negative).grid(row=0, column=0, padx=8, pady=8)
        tb.Button(intensity_tab, text="Auto Contrast Stretch", width=18,
                  bootstyle=SECONDARY, command=self.op_contrast_stretch).grid(row=0, column=1, padx=8, pady=8)

        tb.Label(intensity_tab, text="Gamma (0.1–3.0)").grid(row=1, column=0, padx=8, pady=(12, 4), sticky=W)
        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_scale = tb.Scale(intensity_tab, from_=0.1, to=3.0,
                               orient=tk.HORIZONTAL, variable=self.gamma_var,
                               command=self._on_gamma_change, bootstyle=INFO)
        gamma_scale.grid(row=1, column=1, padx=8, pady=4, sticky=EW)

        intensity_tab.grid_columnconfigure(0, weight=1)
        intensity_tab.grid_columnconfigure(1, weight=1)

        # ===== Filters tab =====
        filters_tab = self.tab_frames["Filters"]

        tb.Button(filters_tab, text="Mean Filter", width=18,
                  bootstyle=SUCCESS, command=self.op_filter_mean_dialog).grid(row=0, column=0, padx=8, pady=8)
        tb.Button(filters_tab, text="Gaussian Filter", width=18,
                  bootstyle=SUCCESS, command=self.op_filter_gaussian_dialog).grid(row=0, column=1, padx=8, pady=8)
        tb.Button(filters_tab, text="Median Filter", width=18,
                  bootstyle=SUCCESS, command=self.op_filter_median_dialog).grid(row=1, column=0, padx=8, pady=8)

        tb.Button(filters_tab, text="Laplacian", width=18,
                  bootstyle=SUCCESS, command=self.op_filter_laplacian).grid(row=1, column=1, padx=8, pady=8)
        tb.Button(filters_tab, text="Sobel X", width=18,
                  bootstyle=SUCCESS, command=lambda: self.op_filter_sobel('x')).grid(row=2, column=0, padx=8, pady=8)
        tb.Button(filters_tab, text="Sobel Y", width=18,
                  bootstyle=SUCCESS, command=lambda: self.op_filter_sobel('y')).grid(row=2, column=1, padx=8, pady=8)

        # Canny Edge Detection button
        tb.Button(filters_tab, text="Canny Edge", width=18,
                  bootstyle=SUCCESS, command=self.op_canny_dialog).grid(row=3, column=0, padx=8, pady=8)

        filters_tab.grid_columnconfigure(0, weight=1)
        filters_tab.grid_columnconfigure(1, weight=1)

        # ===== Hist & Morph tab =====
        hist_tab = self.tab_frames["Hist & Morph"]

        tb.Button(hist_tab, text="Show Histogram", width=18,
                  bootstyle=DANGER, command=self.show_histogram).grid(row=0, column=0, padx=8, pady=8)
        tb.Button(hist_tab, text="Hist Equalization", width=18,
                  bootstyle=DANGER, command=self.op_hist_equalization).grid(row=0, column=1, padx=8, pady=8)

        tb.Button(hist_tab, text="Otsu Threshold", width=18,
                  bootstyle=DANGER, command=self.op_otsu_threshold).grid(row=1, column=0, padx=8, pady=8)
        tb.Button(hist_tab, text="Erode", width=18,
                  bootstyle=DANGER, command=self.op_erode_dialog).grid(row=1, column=1, padx=8, pady=8)
        tb.Button(hist_tab, text="Dilate", width=18,
                  bootstyle=DANGER, command=self.op_dilate_dialog).grid(row=2, column=0, padx=8, pady=8)
        tb.Button(hist_tab, text="Open", width=18,
                  bootstyle=DANGER, command=self.op_open_dialog).grid(row=2, column=1, padx=8, pady=8)
        tb.Button(hist_tab, text="Close", width=18,
                  bootstyle=DANGER, command=self.op_close_dialog).grid(row=3, column=0, padx=8, pady=8)

        hist_tab.grid_columnconfigure(0, weight=1)
        hist_tab.grid_columnconfigure(1, weight=1)

        # Status bar
        status_bar = tb.Label(self, textvariable=self.status_var,
                              anchor=W, bootstyle=SECONDARY)
        status_bar.pack(side=BOTTOM, fill=X, padx=5, pady=3)

        # Initial tab
        self.show_tab("Basics")

    # ---------- Helpers ----------
    def _check_image(self) -> bool:

        if self.current_img is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return False
        return True

    def _update_panels(self):

        if self.original_img is not None:
            pil_orig = cv2_to_pil(self.original_img)
            pil_orig = self._fit_to_panel(pil_orig)
            self.orig_photo = ImageTk.PhotoImage(pil_orig)
            self.left_panel.configure(image=self.orig_photo, text="")
        else:
            self.left_panel.configure(image="", text="Original")

        if self.current_img is not None:
            pil_proc = cv2_to_pil(self.current_img)
            pil_proc = self._fit_to_panel(pil_proc)
            self.proc_photo = ImageTk.PhotoImage(pil_proc)
            self.right_panel.configure(image=self.proc_photo, text="")
        else:
            self.right_panel.configure(image="", text="Result")

    def _fit_to_panel(self, pil_img, max_w=520, max_h=300):

        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        return pil_img.resize(new_size, Image.LANCZOS)

    # ---------- File I/O ----------
    def open_image(self):

        path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*")
        ])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not open image.")
            return

        self.original_img = img
        self.current_img = deepcopy(img)
        h, w = img.shape[:2]
        self.set_status(f"Opened: {os.path.basename(path)}  ({w} x {h})")
        self._update_panels()

    def save_result(self):

        if not self._check_image():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"),
                       ("JPEG", "*.jpg;*.jpeg"),
                       ("BMP", "*.bmp")]
        )
        if not path:
            return

        cv2.imwrite(path, self.current_img)
        self.set_status(f"Saved: {os.path.basename(path)}")
        messagebox.showinfo("Saved", f"Image saved to:\n{path}")

    def reset_image(self):

        if self.original_img is None:
            return
        self.current_img = deepcopy(self.original_img)
        self.gamma_var.set(1.0)
        self.set_status("Image reset to original.")
        self._update_panels()

    # ---------- Basic operations ----------
    def op_to_grayscale(self):
        """Convert current image to grayscale."""
        if not self._check_image():
            return
        if len(self.current_img.shape) == 3:
            self.current_img = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        self.set_status("Converted to grayscale.")
        self._update_panels()

    def op_flip_horizontal(self):
        """Flip the image horizontally."""
        if not self._check_image():
            return
        self.current_img = cv2.flip(self.current_img, 1)
        self.set_status("Flipped horizontally.")
        self._update_panels()

    def op_flip_vertical(self):
        """Flip the image vertically."""
        if not self._check_image():
            return
        self.current_img = cv2.flip(self.current_img, 0)
        self.set_status("Flipped vertically.")
        self._update_panels()

    def op_rotate_90(self, clockwise=True):
        """Rotate the image by 90 degrees."""
        if not self._check_image():
            return
        if clockwise:
            self.current_img = cv2.rotate(self.current_img, cv2.ROTATE_90_CLOCKWISE)
            self.set_status("Rotated 90° clockwise.")
        else:
            self.current_img = cv2.rotate(self.current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.set_status("Rotated 90° counter-clockwise.")
        self._update_panels()

    # ---------- Affine (Rotate / Scale / Translate / Shear) ----------
    def op_affine_rotate_dialog(self):
        """Ask user for an angle and apply affine rotation."""
        if not self._check_image():
            return

        angle = simpledialog.askfloat(
            "Rotate",
            "Enter rotation angle (degrees):",
            parent=self
        )
        if angle is None:
            return

        self.op_affine_rotate(angle)

    def op_affine_rotate(self, angle_degrees: float):

        if not self._check_image():
            return

        img = self.current_img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        angle_norm = angle_degrees % 360.0

        M = cv2.getRotationMatrix2D(center, angle_norm, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        self.current_img = rotated
        self.set_status(
            f"Affine rotate: {angle_norm:.2f}° (input: {angle_degrees:.2f}°)"
        )
        self._update_panels()

    def op_affine_scale_dialog(self):
        """Ask user for a scale factor and apply affine scaling."""
        if not self._check_image():
            return

        factor = simpledialog.askfloat(
            "Scale",
            "Enter scale factor (e.g. 0.5, 2.0):",
            parent=self
        )
        if factor is None:
            return
        if factor <= 0:
            messagebox.showwarning("Invalid factor", "Scale factor must be positive.")
            return

        self.op_affine_scale(factor, factor)

    def op_affine_scale(self, sx: float, sy: float):
        """Scale the image using an affine transform."""
        if not self._check_image():
            return
        img = self.current_img
        h, w = img.shape[:2]

        new_w = int(w * sx)
        new_h = int(h * sy)

        if new_w < MIN_SIZE or new_h < MIN_SIZE:
            messagebox.showwarning("Scale limit",
                                   f"Image is too small to scale further (min {MIN_SIZE}px).")
            self.set_status("Scale blocked: too small.")
            return

        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            messagebox.showwarning("Scale limit",
                                   f"Image would be too large (max {MAX_SIZE}px).")
            self.set_status("Scale blocked: too large.")
            return

        M = np.float32([[sx, 0, 0],
                        [0, sy, 0]])
        scaled = cv2.warpAffine(img, M, (new_w, new_h))
        self.current_img = scaled
        self.set_status(f"Scaled by ({sx:.2f}, {sy:.2f}).")
        self._update_panels()

    def op_affine_translate_dialog(self):

        if not self._check_image():
            return

        dx = simpledialog.askfloat(
            "Translate",
            "Enter dx (pixels):",
            parent=self
        )
        if dx is None:
            return

        dy = simpledialog.askfloat(
            "Translate",
            "Enter dy (pixels):",
            parent=self
        )
        if dy is None:
            return

        self.op_affine_translate(dx, dy)

    def op_affine_translate(self, dx: float, dy: float):

        if not self._check_image():
            return
        img = self.current_img
        h, w = img.shape[:2]
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h))
        self.current_img = shifted
        self.set_status(f"Translated by (dx={dx:.1f}, dy={dy:.1f}).")
        self._update_panels()

    def op_affine_shear_x_dialog(self):

        if not self._check_image():
            return

        shx = simpledialog.askfloat(
            "Shear X",
            "Enter shear factor along X (e.g. 0.3):",
            parent=self
        )
        if shx is None:
            return

        self.op_affine_shear(shx=shx, shy=0.0)

    def op_affine_shear_y_dialog(self):

        if not self._check_image():
            return

        shy = simpledialog.askfloat(
            "Shear Y",
            "Enter shear factor along Y (e.g. 0.3):",
            parent=self
        )
        if shy is None:
            return

        self.op_affine_shear(shx=0.0, shy=shy)

    def op_affine_shear(self, shx: float = 0.0, shy: float = 0.0):
        """Apply an affine shear transform on the image."""
        if not self._check_image():
            return
        img = self.current_img
        h, w = img.shape[:2]

        M = np.float32([[1, shx, 0],
                        [shy, 1, 0]])

        new_w = int(w + abs(shx) * h)
        new_h = int(h + abs(shy) * w)

        if new_w < MIN_SIZE or new_h < MIN_SIZE:
            messagebox.showwarning("Shear limit",
                                   f"Image is too small after shear (min {MIN_SIZE}px).")
            self.set_status("Shear blocked: too small.")
            return

        if new_w > MAX_SIZE or new_h > MAX_SIZE:
            messagebox.showwarning("Shear limit",
                                   f"Image would be too large after shear (max {MAX_SIZE}px).")
            self.set_status("Shear blocked: too large.")
            return

        sheared = cv2.warpAffine(img, M, (new_w, new_h))
        self.current_img = sheared
        self.set_status(f"Shear applied (shx={shx:.2f}, shy={shy:.2f}).")
        self._update_panels()

    # ---------- Intensity ----------
    def op_negative(self):
        """Apply negative transformation."""
        if not self._check_image():
            return
        self.current_img = cv2.bitwise_not(self.current_img)
        self.set_status("Negative applied.")
        self._update_panels()

    def op_contrast_stretch(self):
        """Apply automatic contrast stretching."""
        if not self._check_image():
            return
        img = self.current_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        self.current_img = auto_contrast_stretch(gray)
        self.set_status("Auto contrast stretching applied.")
        self._update_panels()

    def _on_gamma_change(self, _event=None):
        #apply gamma correction.
        if not self._check_image():
            return
        gamma = float(self.gamma_var.get())
        img = self.current_img
        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            b = gamma_correction(b, gamma)
            g = gamma_correction(g, gamma)
            r = gamma_correction(r, gamma)
            out = cv2.merge([b, g, r])
        else:
            out = gamma_correction(img, gamma)
        self.current_img = out
        self.set_status(f"Gamma correction: γ = {gamma:.2f}")
        self._update_panels()

    # ---------- Filters ----------
    def _ensure_gray(self):
        """Return a grayscale version of the current image."""
        img = self.current_img
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _ask_kernel_size(self, title: str):
        """Ask user for an odd kernel size (>=3) and validate."""
        k = simpledialog.askinteger(
            title,
            "Enter kernel size (odd number, >= 3):",
            parent=self,
            minvalue=3,
            maxvalue=99
        )
        if k is None:
            return None

        # Must be odd
        if k % 2 == 0:
            messagebox.showwarning(
                "Invalid kernel size",
                "Kernel size must be an ODD number (3, 5, 7, ...)."
            )
            return None

        return k

    def op_filter_mean_dialog(self):
        """Ask for kernel size and apply mean filter."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Mean Filter")
        if k is None:
            return
        self.op_filter_mean(k)

    def op_filter_gaussian_dialog(self):
        """Ask for kernel size and apply Gaussian filter."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Gaussian Filter")
        if k is None:
            return
        self.op_filter_gaussian(k)

    def op_filter_median_dialog(self):
        """Ask for kernel size and apply median filter."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Median Filter")
        if k is None:
            return
        self.op_filter_median(k)

    def op_filter_mean(self, ksize: int):
        """Apply mean (average) filter."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        blur = cv2.blur(gray, (ksize, ksize))
        self.current_img = blur
        self.set_status(f"Mean filter ({ksize}x{ksize}).")
        self._update_panels()

    def op_filter_gaussian(self, ksize: int):
        """Apply Gaussian filter."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        self.current_img = blur
        self.set_status(f"Gaussian filter ({ksize}x{ksize}).")
        self._update_panels()

    def op_filter_median(self, ksize: int):
        """Apply median filter."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        median = cv2.medianBlur(gray, ksize)
        self.current_img = median
        self.set_status(f"Median filter (k={ksize}).")
        self._update_panels()

    def op_filter_laplacian(self):
        """Apply Laplacian filter for edge detection."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        self.current_img = lap
        self.set_status("Laplacian filter applied.")
        self._update_panels()

    def op_filter_sobel(self, axis='x'):
        """Apply Sobel filter in X or Y direction."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        if axis == 'x':
            sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            self.set_status("Sobel X (ksize=3).")
        else:
            sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            self.set_status("Sobel Y (ksize=3).")
        sob = cv2.convertScaleAbs(sob)
        self.current_img = sob
        self._update_panels()

    # --- Canny Edge Detection ---
    def op_canny_dialog(self):
        """Ask user for min/max thresholds and apply Canny edge detection."""
        if not self._check_image():
            return

        t1 = simpledialog.askinteger(
            "Canny Edge",
            "Enter MIN threshold (0–255):",
            parent=self,
            minvalue=0,
            maxvalue=255
        )
        if t1 is None:
            return

        t2 = simpledialog.askinteger(
            "Canny Edge",
            "Enter MAX threshold (0–255):",
            parent=self,
            minvalue=0,
            maxvalue=255
        )
        if t2 is None:
            return

        if t2 <= t1:
            messagebox.showwarning(
                "Invalid thresholds",
                "MAX threshold must be greater than MIN threshold."
            )
            return

        self.op_canny(t1, t2)

    def op_canny(self, tmin: int, tmax: int):
        """Apply Canny edge detection."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        edges = cv2.Canny(gray, tmin, tmax)
        self.current_img = edges
        self.set_status(f"Canny edge detection applied (min={tmin}, max={tmax}).")
        self._update_panels()

    # ---------- Histogram & Morphology ----------
    def show_histogram(self):
        """Display a histogram window for the current image."""
        if not self._check_image():
            return
        img = self.current_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        win = tb.Toplevel(self)
        win.title("Histogram")

        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(hist)
        ax.set_title("Grayscale Histogram")
        ax.set_xlim([0, 255])

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.set_status("Histogram displayed.")

    def op_hist_equalization(self):
        """Apply histogram equalization to the current image."""
        if not self._check_image():
            return
        img = self.current_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        eq = cv2.equalizeHist(gray)
        self.current_img = eq
        self.set_status("Histogram equalization applied.")
        self._update_panels()

    def op_otsu_threshold(self):
        """Apply Otsu thresholding to obtain a binary image."""
        if not self._check_image():
            return
        gray = self._ensure_gray()
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.current_img = bw
        self.set_status("Otsu thresholding applied.")
        self._update_panels()

    # --- Morphology with kernel dialog ---
    def op_erode_dialog(self):
        """Ask for kernel size and apply erosion."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Erode")
        if k is None:
            return
        self.op_morph('erode', k)

    def op_dilate_dialog(self):
        """Ask for kernel size and apply dilation."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Dilate")
        if k is None:
            return
        self.op_morph('dilate', k)

    def op_open_dialog(self):
        """Ask for kernel size and apply opening."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Open")
        if k is None:
            return
        self.op_morph('open', k)

    def op_close_dialog(self):
        """Ask for kernel size and apply closing."""
        if not self._check_image():
            return
        k = self._ask_kernel_size("Close")
        if k is None:
            return
        self.op_morph('close', k)

    def op_morph(self, mode: str, ksize: int = 3):
        if not self._check_image():
            return
        gray = self._ensure_gray()
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        if mode == 'erode':
            out = cv2.erode(bw, kernel, iterations=1)
            self.set_status(f"Morphology: Erode ({ksize}x{ksize}).")
        elif mode == 'dilate':
            out = cv2.dilate(bw, kernel, iterations=1)
            self.set_status(f"Morphology: Dilate ({ksize}x{ksize}).")
        elif mode == 'open':
            out = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            self.set_status(f"Morphology: Open ({ksize}x{ksize}).")
        elif mode == 'close':
            out = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            self.set_status(f"Morphology: Close ({ksize}x{ksize}).")
        else:
            out = bw
            self.set_status("Morphology: no-op (unknown mode).")

        self.current_img = out
        self._update_panels()


if __name__ == "__main__":
    app = DIPApp()
    app.mainloop()
