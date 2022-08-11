# DustyScans
Scans of empty 35mm film frames with annotated artifacts such as dust, scratches, dirt, hairs. Example of a fully annotated scanned image:

![image](https://github.com/daniela997/DustyScans/blob/main/figures/annotated_example.png)

## Artifact sizes
### Conversion from pixel to micron
The following code converts the measured sizes for each artifact from pixels to microns, assuming that the scanned frames are of standard 35mm by 24mm size.
```
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
```
### Artifact sizes distribution
For all samples, size is measured by calculating the area within the contour via OpenCV, then converting the area from pixels to sq. microns.

![image](https://user-images.githubusercontent.com/32989037/183711634-ef816c59-10cb-4f80-bb24-10b9e559b910.png)

### Examples of extracted artifacts
Each artifact has been padded to square for visualisation purposes.

![image](https://user-images.githubusercontent.com/32989037/183712091-c59b1ac2-985d-49a1-9968-837465d0bf8e.png)
![image](https://user-images.githubusercontent.com/32989037/183712228-d958fdf7-c003-465c-bc53-2562a3c36529.png)
![image](https://user-images.githubusercontent.com/32989037/183712320-0075d557-e1f2-48d0-8eed-27a174369fc4.png)
![image](https://user-images.githubusercontent.com/32989037/183712425-ef32b643-7c71-43f8-a481-251f56f411bc.png)
![image](https://user-images.githubusercontent.com/32989037/183712533-266964b4-56ee-4814-980b-66274c58c85d.png)


## Artifact count and frequency
The annotated artifacts are 12135 in total, across 10 scanned frames. Dust specks are the most common type of artifacts, and scratches are the least common.

![image](https://user-images.githubusercontent.com/32989037/183714488-f04e681f-7318-4d98-a5ab-68aeac497363.png) ![image](https://user-images.githubusercontent.com/32989037/183714598-bb658901-082a-49b5-9daa-4e2f847f0f27.png)

## Artifact spatial frequency
Each scan has been split into 256 by 256 pixel quadrants (with padding where required), in order to record the frequency of each type of artifact for each quadrant.

Screenshot 2022-08-11 at 13.25.34<img width="1242" alt="image" src="https://user-images.githubusercontent.com/32989037/184132982-e45e1c01-890a-4814-ac34-9c823585b6fc.png">

### Spatial distribution of artifacts
![image](https://user-images.githubusercontent.com/32989037/184132734-673fd87d-f825-4baf-a1c0-7073a2425156.png)

