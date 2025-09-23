# viewer.py
import numpy as np
import torch
import pytorch3d
import matplotlib.pyplot as plt

from utils import get_device, get_points_renderer


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
def visualize_pointcloud_interactive(
    point_cloud_path="data/bridge_pointcloud.npz",
    stride: int = 50,
    point_color=None,
    image_size: int = 512,
    background_color=(0, 0, 0),
    device=None,
):
    if device is None:
        device = get_device()

    pc = np.load(point_cloud_path)
    verts = torch.tensor(pc["verts"][::stride], dtype=torch.float32)
    rgb = torch.tensor(pc["rgb"][::stride], dtype=torch.float32)
    if point_color is not None: 
        color = torch.tensor(point_color, dtype=torch.float32, device=device) 
        B, N, _ = verts.shape
        rgb = color.view(1, 1, 3).expand(B, N, 3)

    viewer = P3DPointCloudViewer(
        verts_xyz=verts,
        rgb=rgb,
        image_size=image_size,
        background_color=background_color,
        device=device,
    )
    viewer.show()
