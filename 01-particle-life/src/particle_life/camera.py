"""View transform over the periodic unit-square world. Pure Python."""

from dataclasses import dataclass

ZOOM_MIN, ZOOM_MAX = 1.0, 30.0


@dataclass
class Camera:
    cx: float = 0.5
    cy: float = 0.5
    zoom: float = 1.0

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Screen position in [0,1]^2 (y up) to world coordinates, wrapped to [0,1)."""
        return (
            (self.cx + (sx - 0.5) / self.zoom) % 1.0,
            (self.cy + (sy - 0.5) / self.zoom) % 1.0,
        )

    def pan(self, dx: float, dy: float) -> None:
        self.cx = (self.cx + dx) % 1.0
        self.cy = (self.cy + dy) % 1.0

    def zoom_by(self, factor: float) -> None:
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom * factor))
