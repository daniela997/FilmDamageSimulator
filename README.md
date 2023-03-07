# Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans

![overview](https://user-images.githubusercontent.com/32989037/223543778-a548271f-0cda-493f-91cf-2c38aa5c36cc.png)

## Film Damage Simulator
This repository contains damaged film scans along with their artifact annotations, as used in *"Simulating analogue film damage to analyse and improve artifact restoration on high-resolution scans"*, Eurographics 2023. Additionally, it provides code for our statistical model which uses these extracted artifacts to generate damage overlays for target images in order to produce synthetically damaged training data for the film artifact restoration problem.

## DustyScans
Scans of empty 35mm film frames with annotated artifacts such as dust, scratches, dirt, hairs. Example of a fully annotated scanned image:

![image](https://github.com/daniela997/DustyScans/blob/main/figures/annotated_example.png)
