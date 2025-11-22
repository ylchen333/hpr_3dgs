# viewer.py
import numpy as np
import torch
import pytorch3d
import matplotlib.pyplot as plt

from utils import get_device, get_points_renderer

# viewer.py (near the top of visualize_pointcloud_interactive)

def _to_batched_pNx3(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 2 and x.shape[1] == 3:
        x = x.unsqueeze(0)  # -> (1,N,3)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected (N,3) or (B,N,3), got {tuple(x.shape)}")
    return x

def _prep_colors(c, B, N):
    if c is None:
        return torch.ones((B, N, 3), dtype=torch.float32)
    c = torch.as_tensor(c, dtype=torch.float32)
    if c.ndim == 2 and c.shape[1] == 3:
        c = c.unsqueeze(0)  # (1,N,3)
    if c.ndim != 3 or c.shape[2] not in (3, 4):
        raise ValueError(f"Expected colors (N,3/4) or (B,N,3/4), got {tuple(c.shape)}")
    if c.shape[2] == 4:  # drop alpha if present
        c = c[:, :, :3]
    if c.shape[0] != B or c.shape[1] != N:
        raise ValueError(f"Colors shape {tuple(c.shape)} must match (B,N,3) with B={B}, N={N}")
    # clamp to [0,1]
    return c.clamp(0.0, 1.0)


class P3DPointCloudViewer:
    """
    Controls:
      - Left mouse drag: orbit (azimuth/elevation)
      - Mouse wheel: zoom (distance)
      - R: reset view
      - Q / Esc: close
    """
    def __init__(
        self,
        verts_xyz: torch.Tensor,      # (N, 3) on any device
        rgb: torch.Tensor,            # (N, 3) or (N, 4) in [0,1]
        image_size: int = 512,
        background_color=(0, 0, 0),
        init_distance: float = 4.0,
        init_elev_deg: float = 10.0,
        init_azim_deg: float = 0.0,
        fov: float = 60.0,
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.renderer = get_points_renderer(
            image_size=image_size, background_color=background_color
        )

        # Pack into a PyTorch3D Pointclouds struct
        verts_xyz = verts_xyz.to(self.device)
        rgb = rgb[..., :3].to(self.device)  # ensure RGB
        self.pc = pytorch3d.structures.Pointclouds(
            points=[verts_xyz], features=[rgb]
        )

        # Camera state (spherical)
        self.distance = init_distance
        self.elev = init_elev_deg
        self.azim = init_azim_deg
        self.fov = fov

        # Dragging state
        self._is_dragging = False
        self._last_mouse_xy = None

        # Figure / axes / image
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.axis("off")
        self.im = self.ax.imshow(
            np.zeros((image_size, image_size, 3), dtype=np.float32)
        )
        self.ax.set_title("PyTorch3D Point Cloud Viewer")

        # Connect events
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render_and_draw()

    # -------- rendering ----------
    def _render_and_draw(self):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=self.distance, elev=self.elev, azim=self.azim, device=self.device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, device=self.device, fov=self.fov
        )
        with torch.no_grad():
            img = self.renderer(self.pc, cameras=cameras)[0, ..., :3].detach().cpu().numpy()
        self.im.set_data(np.clip(img, 0, 1))
        self.fig.canvas.draw_idle()

    # -------- interaction ----------
    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # left button = orbit
            self._is_dragging = True
            self._last_mouse_xy = (event.x, event.y)

    def _on_release(self, event):
        self._is_dragging = False
        self._last_mouse_xy = None

    def _on_motion(self, event):
        if not self._is_dragging or event.inaxes != self.ax:
            return
        if self._last_mouse_xy is None:
            self._last_mouse_xy = (event.x, event.y)
            return

        x0, y0 = self._last_mouse_xy
        dx = event.x - x0
        dy = event.y - y0
        self._last_mouse_xy = (event.x, event.y)

        # Sensitivities (tweak to taste)
        azim_sens = 0.3
        elev_sens = 0.3

        self.azim = (self.azim - dx * azim_sens) % 360.0
        self.elev = np.clip(self.elev + dy * elev_sens, -89.9, 89.9)
        self._render_and_draw()

    def _on_scroll(self, event):
        # Zoom in/out by changing camera distance
        zoom_factor = 0.9 if event.button == "up" else 1.1
        self.distance = float(np.clip(self.distance * zoom_factor, 0.5, 50.0))
        self._render_and_draw()

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key.lower() == "r":
            self.distance = 4.0
            self.elev = 10.0
            self.azim = 0.0
            self.fov = 60.0
            self._render_and_draw()

    def show(self):
        plt.show()


# ---------- convenience function that plugs into your code ----------
# from viewer import P3DPointCloudViewer  # wherever it's defined

def visualize_pointcloud_interactive(
    point_cloud_path="data/bridge_pointcloud.npz",
    stride: int = 50,
    point_color=None,                 # e.g., (1.0, 0.2, 0.2)
    image_size: int = 512,
    background_color=(0, 0, 0),
    device=None,
):
    if device is None:
        device = get_device()

    # --- load NPZ ---
    data = np.load(point_cloud_path)

    print("viewer verts", data["verts"].shape, data["verts"].dtype)
    print("viewer rgb",   data["rgb"].shape,   data["rgb"].dtype)
    if "verts" not in data:
        raise ValueError(f"'verts' not found in {point_cloud_path}")
    verts_np = data["verts"]
    rgb_np   = data["rgb"] if "rgb" in data.files else np.ones_like(verts_np, dtype=np.float32)

    # --- stride / subsample ---
    verts_np = verts_np[::stride]
    rgb_np   = rgb_np[::stride]

    # --- drop alpha if present, normalize to [0,1] ---
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[:, :3]
    rgb_np = rgb_np.astype(np.float32)
    if rgb_np.max() > 1.0:
        rgb_np /= 255.0
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    # --- override color if requested ---
    if point_color is not None:
        color = np.asarray(point_color, dtype=np.float32).reshape(1, 3)
        rgb_np = np.repeat(color, repeats=verts_np.shape[0], axis=0)

    # --- to torch, add batch dim -> (1,N,3) ---
    verts = torch.from_numpy(verts_np).float()
    rgb   = torch.from_numpy(rgb_np).float()
    # if verts.ndim == 2:
    #     verts = verts.unsqueeze(0)
    # if rgb.ndim == 2:
    #     rgb = rgb.unsqueeze(0)

    # --- final sanity checks ---
    if verts.shape[-1] != 3 or rgb.shape[-1] not in (3,):
        raise ValueError(f"Expected verts/rgb last dim = 3, got {verts.shape} / {rgb.shape}")
    if verts.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Verts {verts.shape} and RGB {rgb.shape} must have same (B,N)")

    # --- to device ---
    verts = verts.to(device)
    rgb   = rgb.to(device)

    # --- call your viewer (expects batched tensors) ---
    viewer = P3DPointCloudViewer(
        verts_xyz=verts,               # (B,N,3)
        rgb=rgb,                       # (B,N,3)
        image_size=image_size,
        background_color=background_color,
        device=device,
    )
    viewer.show()


