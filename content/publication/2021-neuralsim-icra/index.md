---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "NeuralSim: Augmenting Differentiable Simulators with Neural Networks"
authors:
- "Eric Heiden"
- "David Millard"
- "Erwin Coumans"
- "Yizhou Sheng"
- "Gaurav S. Sukhatme"
author_notes:
- "Equal contribution"
- "Equal contribution"
date: 2020-11-09
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2020-11-09T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "Submitted to International Conference on Robotics and Automation (ICRA) 2021"
publication_short: ""

abstract: "Differentiable simulators provide an avenue for closing the sim-to-real gap by enabling the use of efficient, gradient-based optimization algorithms to find the simulation parameters that best fit the observed sensor readings. Nonetheless, these analytical models can only predict the dynamical behavior of systems for which they have been designed. In this work, we study the augmentation of a novel differentiable rigid-body physics engine via neural networks that is able to learn nonlinear relationships between dynamic quantities and can thus learn effects not accounted for in traditional simulators.Such augmentations require less data to train and generalize better compared to entirely data-driven models. Through extensive experiments, we demonstrate the ability of our hybrid simulator to learn complex dynamics involving frictional contacts from real data, as well as match known models of viscous friction, and present an approach for automatically discovering useful augmentations. We show that, besides benefiting dynamics modeling, inserting neural networks can accelerate model-based control architectures. We observe a ten-fold speed-up when replacing the QP solver inside a model-predictive gait controller for quadruped robots with a neural network, allowing us to significantly improve control delays as we demonstrate in real-hardware experiments."

# Summary. An optional shortened abstract.
summary: ""

tags: []
categories: []
featured: true

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_pdf: https://arxiv.org/abs/2011.04217
url_code: https://github.com/google-research/tiny-differentiable-simulator
url_dataset:
url_poster:
url_project: https://sites.google.com/usc.edu/neuralsim
url_slides:
url_source:
url_video: https://youtu.be/hGphZbMDbYA

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Golf simulation augmented by a neural network to learn air resistance"
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
