Each JSON files corresponds to the scanned image with the same name in the `scans` folder. The annotations were manually created and include the following classes: 
* dust
* dirt
* long hair
* short hair
* scratch

Each annotation is a list of point coordinates which can easily be transformed to an OpenCV contour format.

NB: The annotations for scans 8, 9 and 10 need to be rescaled by factor of 1.5 after being loaded as the annotation was done on lower resolution versions of the scans.
