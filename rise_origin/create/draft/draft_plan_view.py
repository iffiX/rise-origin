# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The draft plan view: a CAD style 2D plan of one queried box of space.

The view reads cover boxes only, seen from above. It answers who is here
(boxes and clusters), shows it for the eye (a labeled plan image), and says
it for an agent (deterministic plain-text descriptions). Coordinates are the
frame of the draft that made the view.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import STRtree, box as shapely_box

from ..data.box import AABB

if TYPE_CHECKING:
    from .draft import Draft

__all__ = ["DraftPlanViewBox", "DraftPlanViewCluster", "DraftPlanView"]

_MIN_GRID_SPACING_PX = 40
_LABEL_SIZES = (14, 10)
_AXIS_LABEL_SIZE = 20
_MARGIN_PX = 22
_TAG_SIZE = 10
_TAG_PAD_PX = 3


@dataclass(frozen=True)
class DraftPlanViewBox:
    """One cover box intersecting the queried region, seen from above.

    Attributes:
        node: The draft the box covers.
        min_xy: Lower corner of the footprint in the view frame.
        max_xy: Upper corner of the footprint in the view frame.
        z_interval: The box's z range in the view frame.
        is_cut: Whether the band's top plane passes through the box;
            other boxes lie inside or below the band.
    """

    node: "Draft"
    min_xy: Tuple[float, float]
    max_xy: Tuple[float, float]
    z_interval: Tuple[float, float]
    is_cut: bool


@dataclass(frozen=True)
class DraftPlanViewCluster:
    """One connected clump of footprints with its member nodes.

    Attributes:
        bounds: The clump's bounds in the frame of the draft that made
            the view.
        nodes: The nodes whose footprints touch into this clump.
    """

    bounds: AABB
    nodes: Tuple["Draft", ...]


class DraftPlanView:
    """A CAD style 2D plan over one box of space, covers only.

    Args:
        boxes: The intersecting cover boxes, in the frame of the draft
            that made the view.
        min_bound: Lower corner of the queried box, same frame as boxes.
        max_bound: Upper corner of the queried box, same frame as boxes.
    """

    def __init__(
        self,
        boxes: Sequence[DraftPlanViewBox],
        min_bound: Tuple[float, float, float],
        max_bound: Tuple[float, float, float],
    ) -> None:
        self.boxes = list(boxes)
        self.min_bound = tuple(float(v) for v in min_bound)
        self.max_bound = tuple(float(v) for v in max_bound)
        self._tree: Optional[STRtree] = None

    def get_bounds(self) -> AABB:
        """Return the tight bounds of the actual content, view frame.

        The query box may be huge while the content is small; this is what
        you really have.
        """
        if not self.boxes:
            raise ValueError("the view holds no content")
        lowers = np.array(
            [[b.min_xy[0], b.min_xy[1], b.z_interval[0]] for b in self.boxes]
        )
        uppers = np.array(
            [[b.max_xy[0], b.max_xy[1], b.z_interval[1]] for b in self.boxes]
        )
        return AABB(tuple(lowers.min(axis=0)), tuple(uppers.max(axis=0)))

    def get_sub_clusters(self) -> List[DraftPlanViewCluster]:
        """Return the content clustered into connected clumps of footprints."""
        if not self.boxes:
            return []
        tree = self._get_tree()
        parents = list(range(len(self.boxes)))
        geoms = [
            shapely_box(b.min_xy[0], b.min_xy[1], b.max_xy[0], b.max_xy[1])
            for b in self.boxes
        ]
        for index, geom in enumerate(geoms):
            for other in tree.query(geom, predicate="intersects"):
                DraftPlanView._join(parents, index, int(other))
        groups: Dict[int, List[DraftPlanViewBox]] = {}
        for index, plan_box in enumerate(self.boxes):
            groups.setdefault(DraftPlanView._find(parents, index), []).append(plan_box)
        clusters = []
        for members in groups.values():
            lowers = np.array(
                [[b.min_xy[0], b.min_xy[1], b.z_interval[0]] for b in members]
            )
            uppers = np.array(
                [[b.max_xy[0], b.max_xy[1], b.z_interval[1]] for b in members]
            )
            nodes = []
            for member in members:
                if member.node not in nodes:
                    nodes.append(member.node)
            nodes.sort(key=lambda node: (node.name, node.uid))
            clusters.append(
                DraftPlanViewCluster(
                    AABB(tuple(lowers.min(axis=0)), tuple(uppers.max(axis=0))),
                    tuple(nodes),
                )
            )
        clusters.sort(key=lambda cluster: (cluster.bounds.min.x, cluster.bounds.min.y))
        return clusters

    def to_image(
        self, width: int = 1000, grid_resolution: Optional[float] = None
    ) -> Image.Image:
        """Render the plan for the eye and return a PIL image.

        Boxes paint large first as filled rects colored by a stable uid
        hash; labels center when they fit and shrink one step. A box too
        small for any label gets a name tag placed beside it, tied to the
        box by a dashed leader line. Meter grid lines, a scale bar spanning
        a whole number of grid cells, and the band z range complete the
        plan. The x and y axes are labeled in meters and the z range of
        the viewed band sits in the title, top right, so nothing reads as a
        third axis. The image shows inline in an ipykernel and saves with
        ``image.save(path)``.

        Args:
            width: Image width in pixels; the height follows the view.
            grid_resolution: Grid step in meters; None picks the smallest
                1, 2, or 5 decade step that keeps lines readable.
        """
        extent_x = max(self.max_bound[0] - self.min_bound[0], 1e-9)
        extent_y = max(self.max_bound[1] - self.min_bound[1], 1e-9)
        scale = (width - 2 * _MARGIN_PX) / extent_x
        plot_height = int(round(extent_y * scale))
        step = grid_resolution or DraftPlanView._make_step_meters(
            scale, _MIN_GRID_SPACING_PX
        )
        height = plot_height + 2 * _MARGIN_PX + 24
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        self._paint_grid(draw, scale, plot_height, width, step)
        tags = self._paint_boxes(draw, scale, plot_height)
        self._paint_tags(draw, tags, width, height)
        self._paint_origin(draw, scale, plot_height)
        self._paint_marks(draw, scale, plot_height, width, step)
        return image

    def to_description(self) -> str:
        """Return one deterministic plain sentence per box.

        Each line names the box, its footprint size, its center, its z
        interval, and whether the band's top plane cuts it.
        """
        lines = []
        for plan_box in self._get_ordered_boxes():
            size_x = plan_box.max_xy[0] - plan_box.min_xy[0]
            size_y = plan_box.max_xy[1] - plan_box.min_xy[1]
            center_x = (plan_box.max_xy[0] + plan_box.min_xy[0]) * 0.5
            center_y = (plan_box.max_xy[1] + plan_box.min_xy[1]) * 0.5
            state = "cut by the band top" if plan_box.is_cut else "below the band top"
            lines.append(
                f"{plan_box.node.name}: {size_x:.2f} by {size_y:.2f} m at "
                f"({center_x:.2f}, {center_y:.2f}), z {plan_box.z_interval[0]:.2f} "
                f"to {plan_box.z_interval[1]:.2f} m, {state}."
            )
        return "\n".join(lines)

    def to_sub_clusters_description(self) -> str:
        """Return one deterministic plain sentence per cluster."""
        lines = []
        for number, cluster in enumerate(self.get_sub_clusters(), start=1):
            names = ", ".join(node.name for node in cluster.nodes)
            lo, hi = cluster.bounds.min, cluster.bounds.max
            lines.append(
                f"cluster {number}: from ({lo.x:.2f}, {lo.y:.2f}) to "
                f"({hi.x:.2f}, {hi.y:.2f}) m holding {names}."
            )
        return "\n".join(lines)

    def _get_ordered_boxes(self) -> List[DraftPlanViewBox]:
        """Return the boxes in the stable description order."""
        return sorted(
            self.boxes,
            key=lambda b: (b.node.name, b.node.uid, b.min_xy, b.max_xy),
        )

    def _get_painted_order(self) -> List[DraftPlanViewBox]:
        """Return the boxes largest footprint first for painting."""
        return sorted(
            self._get_ordered_boxes(),
            key=lambda b: -((b.max_xy[0] - b.min_xy[0]) * (b.max_xy[1] - b.min_xy[1])),
        )

    def _make_pixel_rect(
        self, plan_box: DraftPlanViewBox, scale: float, plot_height: int
    ) -> Tuple[int, int, int, int]:
        """Return the pixel rectangle of one box, y flipped for the image."""
        x0 = _MARGIN_PX + (plan_box.min_xy[0] - self.min_bound[0]) * scale
        x1 = _MARGIN_PX + (plan_box.max_xy[0] - self.min_bound[0]) * scale
        y0 = _MARGIN_PX + plot_height - (plan_box.max_xy[1] - self.min_bound[1]) * scale
        y1 = _MARGIN_PX + plot_height - (plan_box.min_xy[1] - self.min_bound[1]) * scale
        return int(x0), int(y0), int(x1), int(y1)

    def _paint_grid(
        self,
        draw: ImageDraw.ImageDraw,
        scale: float,
        plot_height: int,
        width: int,
        step: float,
    ) -> None:
        """Paint the meter grid lines and tick labels behind the boxes."""
        color = (175, 175, 175)
        tick_color = (60, 60, 60)
        font = DraftPlanView._make_font(_TAG_SIZE)
        bottom = _MARGIN_PX + plot_height
        x = np.ceil(self.min_bound[0] / step) * step
        while x <= self.max_bound[0]:
            px = int(_MARGIN_PX + (x - self.min_bound[0]) * scale)
            DraftPlanView._paint_dashed_line(
                draw, (px, _MARGIN_PX), (px, bottom), color
            )
            draw.line([(px, bottom), (px, bottom + 3)], fill=tick_color)
            text = f"{x:g}"
            text_w = int(len(text) * _TAG_SIZE * 0.62)
            draw.text((px - text_w // 2, bottom + 4), text, font=font, fill=tick_color)
            x += step
        y = np.ceil(self.min_bound[1] / step) * step
        while y <= self.max_bound[1]:
            py = int(_MARGIN_PX + plot_height - (y - self.min_bound[1]) * scale)
            DraftPlanView._paint_dashed_line(
                draw, (_MARGIN_PX, py), (width - _MARGIN_PX, py), color
            )
            draw.line([(_MARGIN_PX - 3, py), (_MARGIN_PX, py)], fill=tick_color)
            text = f"{y:g}"
            text_w = int(len(text) * _TAG_SIZE * 0.62)
            draw.text(
                (_MARGIN_PX - 5 - text_w, py - 5), text, font=font, fill=tick_color
            )
            y += step

    def _paint_boxes(
        self, draw: ImageDraw.ImageDraw, scale: float, plot_height: int
    ) -> List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]:
        """Paint the filled labeled rects; return the boxes needing tags."""
        tags = []
        for plan_box in self._get_painted_order():
            color = DraftPlanView._make_color(plan_box.node.uid)
            x0, y0, x1, y1 = self._make_pixel_rect(plan_box, scale, plot_height)
            outline = (60, 60, 60) if plan_box.is_cut else (150, 150, 150)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline)
            name = plan_box.node.name
            size = DraftPlanView._check_label_fits(name, x1 - x0, y1 - y0)
            if size:
                font = DraftPlanView._make_font(size)
                text_color = DraftPlanView._make_text_color(color)
                draw.text(
                    ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
                    name,
                    font=font,
                    fill=text_color,
                    anchor="mm",
                )
            else:
                tags.append(((x0, y0, x1, y1), name, color))
        return tags

    def _paint_tags(
        self,
        draw: ImageDraw.ImageDraw,
        tags: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]],
        width: int,
        height: int,
    ) -> None:
        """Paint a name tag beside each too-small box, tied by a dashed line.

        Tag spots try eight fixed offsets around the box and take the first
        one inside the image that misses every tag placed before, so the
        layout is deterministic.
        """
        font = DraftPlanView._make_font(_TAG_SIZE)
        placed: List[Tuple[int, int, int, int]] = []
        for (x0, y0, x1, y1), name, color in tags:
            text_w = int(len(name) * _TAG_SIZE * 0.62) + 2 * _TAG_PAD_PX
            text_h = _TAG_SIZE + 2 * _TAG_PAD_PX
            anchor = (x1, y0)
            spots = [
                (x1 + 10, y0 - text_h - 6),
                (x1 + 10, y1 + 6),
                (x0 - text_w - 10, y0 - text_h - 6),
                (x0 - text_w - 10, y1 + 6),
                (x1 + 18, y0 - text_h - 18),
                (x1 + 18, y1 + 18),
                (x0 - text_w - 18, y0 - text_h - 18),
                (x0 - text_w - 18, y1 + 18),
            ]
            spot = spots[0]
            for candidate in spots:
                rect = (
                    candidate[0],
                    candidate[1],
                    candidate[0] + text_w,
                    candidate[1] + text_h,
                )
                is_inside = (
                    rect[0] >= 0
                    and rect[1] >= 0
                    and rect[2] <= width
                    and rect[3] <= height
                )
                is_clear = all(
                    rect[2] < held[0]
                    or held[2] < rect[0]
                    or rect[3] < held[1]
                    or held[3] < rect[1]
                    for held in placed
                )
                if is_inside and is_clear:
                    spot = candidate
                    break
            rect = (spot[0], spot[1], spot[0] + text_w, spot[1] + text_h)
            placed.append(rect)
            center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
            DraftPlanView._paint_dashed_line(draw, anchor, center, (90, 90, 90))
            draw.rectangle(rect, fill=(255, 255, 255), outline=color)
            draw.text(
                (rect[0] + _TAG_PAD_PX, rect[1] + _TAG_PAD_PX - 1),
                name,
                font=font,
                fill=(0, 0, 0),
            )

    def _paint_origin(
        self, draw: ImageDraw.ImageDraw, scale: float, plot_height: int
    ) -> None:
        """Mark the frame origin with a cross when it sits inside the view."""
        px = _MARGIN_PX + (0.0 - self.min_bound[0]) * scale
        py = _MARGIN_PX + plot_height - (0.0 - self.min_bound[1]) * scale
        if not (
            _MARGIN_PX
            <= px
            <= _MARGIN_PX + (self.max_bound[0] - self.min_bound[0]) * scale
            and _MARGIN_PX <= py <= _MARGIN_PX + plot_height
        ):
            return
        color = (40, 40, 40)
        draw.line([(px - 7, py), (px + 7, py)], fill=color, width=2)
        draw.line([(px, py - 7), (px, py + 7)], fill=color, width=2)
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], outline=color)
        draw.text(
            (px + 9, py + 3),
            "(0, 0)",
            font=DraftPlanView._make_font(_AXIS_LABEL_SIZE),
            fill=color,
        )

    def _paint_marks(
        self,
        draw: ImageDraw.ImageDraw,
        scale: float,
        plot_height: int,
        width: int,
        step: float,
    ) -> None:
        """Paint the title, axis labels, axis gadget, and scale bar."""
        color = (0, 0, 0)
        label_font = DraftPlanView._make_font(_AXIS_LABEL_SIZE)
        title = (
            f"plan view (x-y)   z range {self.min_bound[2]:.2f} "
            f"to {self.max_bound[2]:.2f} m"
        )
        title_w = draw.textlength(title, font=label_font)
        draw.text((width - _MARGIN_PX - title_w, 1), title, font=label_font, fill=color)
        draw.text((4, 1), "y (m)", font=label_font, fill=color)
        x_label = "x (m)"
        x_label_w = draw.textlength(x_label, font=label_font)
        draw.text(
            (width // 2 - x_label_w // 2, _MARGIN_PX + plot_height + 18),
            x_label,
            font=label_font,
            fill=color,
        )
        base_x = _MARGIN_PX + 14
        base_y = _MARGIN_PX + plot_height - 14
        reach = 30
        draw.line([(base_x, base_y), (base_x + reach, base_y)], fill=color, width=2)
        draw.line(
            [(base_x + reach, base_y), (base_x + reach - 5, base_y - 4)], fill=color
        )
        draw.line(
            [(base_x + reach, base_y), (base_x + reach - 5, base_y + 4)], fill=color
        )
        draw.line([(base_x, base_y), (base_x, base_y - reach)], fill=color, width=2)
        draw.line(
            [(base_x, base_y - reach), (base_x - 4, base_y - reach + 5)], fill=color
        )
        draw.line(
            [(base_x, base_y - reach), (base_x + 4, base_y - reach + 5)], fill=color
        )
        draw.text((base_x + reach + 5, base_y - 11), "x", font=label_font, fill=color)
        draw.text((base_x - 5, base_y - reach - 24), "y", font=label_font, fill=color)
        cells = max(1, int(np.ceil(width / 8 / max(step * scale, 1e-9))))
        bar_meters = cells * step
        bar_px = int(bar_meters * scale)
        base_y = _MARGIN_PX + plot_height + 22
        bar_end = width - _MARGIN_PX
        bar_start = bar_end - bar_px
        draw.line([(bar_start, base_y), (bar_end, base_y)], fill=(0, 0, 0), width=2)
        bar_text = f"{bar_meters:g} m"
        text_w = draw.textlength(bar_text, font=label_font)
        draw.text(
            (bar_start - text_w - 8, base_y - 11),
            bar_text,
            font=label_font,
            fill=(0, 0, 0),
        )

    def _get_tree(self) -> STRtree:
        """Return the 2D tree over the footprints, bulk-built once."""
        if self._tree is None:
            self._tree = STRtree(
                [
                    shapely_box(b.min_xy[0], b.min_xy[1], b.max_xy[0], b.max_xy[1])
                    for b in self.boxes
                ]
            )
        return self._tree

    @staticmethod
    def _make_step_meters(scale: float, min_px: float) -> float:
        """Return the smallest 1, 2, or 5 decade step spanning enough pixels."""
        target = min_px / max(scale, 1e-12)
        power = 10.0 ** np.floor(np.log10(target))
        for factor in (1.0, 2.0, 5.0, 10.0):
            step = factor * power
            if step * scale >= min_px:
                return step
        return 10.0 * power

    @staticmethod
    def _paint_dashed_line(
        draw: ImageDraw.ImageDraw,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Paint a dashed line by alternating short strokes and gaps."""
        length = max(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5, 1e-9)
        step_x = (end[0] - start[0]) / length
        step_y = (end[1] - start[1]) / length
        travel = 0.0
        while travel < length:
            stop = min(travel + 4.0, length)
            draw.line(
                [
                    (start[0] + step_x * travel, start[1] + step_y * travel),
                    (start[0] + step_x * stop, start[1] + step_y * stop),
                ],
                fill=color,
            )
            travel = stop + 3.0

    @staticmethod
    def _check_label_fits(name: str, width: float, height: float) -> Optional[int]:
        """Return the largest label size fitting a rect, or None."""
        for size in _LABEL_SIZES:
            if len(name) * size * 0.62 <= width * 0.92 and size <= height * 0.8:
                return size
        return None

    @staticmethod
    def _make_font(size: int) -> ImageFont.ImageFont:
        """Return the default font at a size, falling back to one size."""
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _make_color(uid: int) -> Tuple[int, int, int]:
        """Return a stable readable color for a node uid."""
        value = (uid * 2654435761) & 0xFFFFFF
        red = ((value >> 16) & 0xFF) // 2 + 64
        green = ((value >> 8) & 0xFF) // 2 + 64
        blue = (value & 0xFF) // 2 + 64
        return red, green, blue

    @staticmethod
    def _make_text_color(fill: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Return black or white by the fill's brightness."""
        brightness = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        return (0, 0, 0) if brightness > 128 else (255, 255, 255)

    @staticmethod
    def _find(parents: List[int], index: int) -> int:
        """Return the union-find root of an index."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    @staticmethod
    def _join(parents: List[int], a: int, b: int) -> None:
        """Union two indexes."""
        root_a = DraftPlanView._find(parents, a)
        root_b = DraftPlanView._find(parents, b)
        if root_a != root_b:
            parents[root_b] = root_a
