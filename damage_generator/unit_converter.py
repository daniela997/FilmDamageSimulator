from typing import Tuple

class PixelToMicron():
    def __init__(self, scan_width, scan_height) -> None:
        self.frame_width_microns = 35 * 1000.0
        self.frame_height_microns = 24 * 1000.0
        self.scan_width = scan_width
        self.scan_height = scan_height
        self.width_ratio = self.frame_width_microns / self.scan_width 
        self.height_ratio = self.frame_height_microns / self.scan_height

    def convert_unit(self, bbox_width, bbox_height) -> Tuple[float, float]:
        bbox_width_microns = bbox_width * self.width_ratio 
        bbox_height_microns = bbox_height * self.height_ratio 
        return bbox_width_microns, bbox_height_microns
    
    def convert_area(self, area_pixels) -> float:
        return area_pixels * self.width_ratio * self.height_ratio