+++
title = "Indian Flora Project"
date = 2018-07-10T01:27:06+05:30
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["machine-learning", "deep-learning", "neural-networks", "research", "internship", "plant-identification", "disease-detection", "computer-vision"]

# Project summary to display on homepage.
summary = "Social Image Data Based Plant Species Identification and Disease Detection"

# Optional image to display on homepage.
image_preview = "indianflora.jpg"

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Does the project detail page use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = ""
caption = ""

+++
![Plant Collage](../../img/indianflora.jpg)

* The full project report can be found [here](https://drive.google.com/file/d/1hhzZYk9c4zbqJtzZ46NwfAZz0b-7W56j/view?usp=sharing).
* Employed Convolutional Neural Network model (ResNet18) by iterating through several different models
trained from-the-scratch, fine-tuned on ImageNet, PlantCLEF Encyclopedia of Life, PlantCLEF noisy web
datasets on a manually collected, robust dataset of 100 Indian plant species. (using PyTorch)
* Experimental observations show efficient computation time and high top-5 precision of 99.85% compared to
the low performance of state-of-the-art approaches that focus on hand-engineered features.
* Deployed a collaborative web-portal using Angular and NodeJS and an Android mobile application where
users can add new observations to the catalogue and query for species identification.
* *The research was supported by the Indian Academy of Sciences’ Summer Research Fellowship.*