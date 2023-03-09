# Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans

[Project Page](https://daniela997.github.io/FilmDamageSimulator/) [Dataset 1](https://doi.org/10.6084/m9.figshare.21803304.v2) [Dataset 2](https://doi.org/10.6084/m9.figshare.21803292)


![overview](https://user-images.githubusercontent.com/32989037/223543778-a548271f-0cda-493f-91cf-2c38aa5c36cc.png)

## Film Damage Simulator
This repository contains damaged film scans along with their artifact annotations, as used in *"Simulating analogue film damage to analyse and improve artifact restoration on high-resolution scans"*, Eurographics 2023. Additionally, it provides code for our statistical model which uses these extracted artifacts to generate damage overlays for target images in order to produce synthetically damaged training data for the film artifact restoration problem. The code can be found in `src` in the `main` branch.
