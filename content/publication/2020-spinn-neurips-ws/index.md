---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Sparse-Input Neural Network Augmentations for Differentiable Simulators"
authors:
- "Eric Heiden"
- "David Millard"
- "Erwin Coumans"
- "Gaurav S. Sukhatme"

date: 2020-11-09
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2020-11-09T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["4"]

# Publication name and optional abbreviated publication name.
publication: "NeurIPS 2020 [Workshop: Differentiable computer vision, graphics, and physics in machine learning](https://montrealrobotics.ca/diffcvgp/)"
publication_short: "DiffCVGP Workshop"

abstract: "Differentiable simulators provide an avenue for closing the sim2real gap by enabling the use of efficient, gradient-based optimization algorithms to find the simulation parameters that best fit the observed sensor readings.
Nonetheless, these analytical models can only predict the dynamical behavior of systems they have been designed for.
In this work, we study the augmentation of a differentiable rigid-body physics engine via neural networks that is able to learn nonlinear relationships between dynamic quantities and can thus learn effects not accounted for in the traditional simulator. By optimizing with a sparse-group lasso penalty, we are able to reduce the neural augmentations to the necessary inputs that are needed in the learned models of the simulator while achieving highly accurate predictions."

# Summary. An optional shortened abstract.
summary: ""

tags: []
categories: []
featured: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_pdf:
url_code: https://github.com/google-research/tiny-differentiable-simulator
url_dataset:
url_poster: poster.pdf
url_project: https://sites.google.com/usc.edu/neuralsim
url_slides:
url_source:
url_video: https://youtu.be/hGphZbMDbYA

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: ["ids"]

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---

<img src=training.gif/>
