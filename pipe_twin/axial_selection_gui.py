"""Mouse interaction for an image direction and its perpendicular end limits."""
from __future__ import annotations

import numpy as np

from .axial_selection import normalize_axial_band, move_band_limit, band_overlay


class AxialBandController:
    def __init__(self, view):
        self.view = view
        self.start_point = None
        self.original = None
        self.limit_index = None
        self.draft = None

    @property
    def dragging(self):
        return self.start_point is not None or self.limit_index is not None

    def pixel(self, event, *, clamp=True):
        scale, ox, oy = self.view.transform
        point = np.array([(event.x-ox)/scale, (event.y-oy)/scale])
        size = np.array(self.view.image.shape[1::-1])
        if not clamp and (np.any(point < 0) or np.any(point > size-1)):
            return None
        return np.clip(point, 0, size-1).tolist()

    def start(self, event):
        owner = self.view.owner
        if owner.busy or self.view.image is None or owner.region_selection_active:
            return False
        point = self.pixel(event, clamp=False)
        if point is None:
            return False
        if owner.axial_selection_active:
            self.start_point = point
            self.view.canvas.focus_set()
            return True
        band = owner.registration_settings.get('aggregation_band')
        if not band:
            return False
        scale = self.view.transform[0]
        distances = np.linalg.norm(np.asarray(band['axis_segment_px'])-point, axis=1)*scale
        index = int(np.argmin(distances))
        if distances[index] <= 12:
            self.original = band
            self.limit_index = index
            self.view.canvas.focus_set()
            return True
        return False

    def move(self, event):
        if not self.dragging:
            return False
        if self.view.owner.busy:
            self.cancel()
            return True
        point = self.pixel(event)
        try:
            if self.limit_index is not None:
                self.draft = move_band_limit(self.original, self.limit_index, point)
            else:
                self.draft = normalize_axial_band({'image_size': list(self.view.image.shape[1::-1]),
                                                   'axis_segment_px': [self.start_point, point]})
        except ValueError:
            self.draft = None
        self.draw()
        return True

    def finish(self, event):
        if not self.dragging:
            return False
        self.move(event)
        draft = self.draft
        self.cancel()
        if draft is None:
            self.view.owner.message.set('方向线太短，请沿管长拖出至少 32 个原图像素；建议汇聚范围达到 384 px。')
        else:
            self.view.owner.set_axial_band(draft)
        self.draw()
        return True

    def cancel(self):
        self.start_point = self.original = self.limit_index = self.draft = None

    def draw(self):
        canvas = self.view.canvas
        canvas.delete('axial-band')
        band = self.draft or self.view.owner.registration_settings.get('aggregation_band')
        if band is None or self.view.image is None:
            return
        if tuple(band['image_size']) != self.view.image.shape[1::-1]:
            return
        geometry = band_overlay(band)
        scale, ox, oy = self.view.transform

        def screen(points):
            return (np.asarray(points)*scale+[ox, oy]).ravel().tolist()

        if len(geometry['polygon_px']) >= 3:
            canvas.create_polygon(*screen(geometry['polygon_px']), fill='#42BFC2', stipple='gray12',
                                  outline='', tags=('axial-band', 'axial-band-area'))
        for line in geometry['boundary_lines_px']:
            if len(line) == 2:
                canvas.create_line(*screen(line), fill='#FFB347', width=2,
                                   tags=('axial-band', 'axial-band-limit'))
        canvas.create_line(*screen(band['axis_segment_px']), fill='#FFE07A', width=2, arrow='last',
                           dash=(5, 3), tags=('axial-band', 'axial-band-axis'))
        for index, point in enumerate(band['axis_segment_px']):
            x, y = screen([point])
            canvas.create_oval(x-6, y-6, x+6, y+6, fill='#FFB347', outline='#162B40', width=1,
                               tags=('axial-band', 'axial-band-handle'))
            canvas.create_text(x+10, y-12, text=f'边界 {index+1}', fill='#FFE07A', anchor='w',
                               tags='axial-band')
        center = np.asarray(band['axis_segment_px']).mean(axis=0)
        x, y = screen([center])
        canvas.create_text(x, y+18, text=f"汇聚范围 {geometry['span_px']:.0f} px", fill='#FFE07A',
                           tags='axial-band')
