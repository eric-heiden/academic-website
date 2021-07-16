---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Interactive Differentiable Simulation"
summary: "Reduce reality gap in robotics through differentiable simulators which are updated continuously from real-world measurements"
authors: []
tags: []
categories: []
date: 2019-10-29T16:19:16-07:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: true

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

<img align="right" src="featured.png" width="200"/>

I am investigating how differentiable simulators can help us reduce the simulation-reality gap. The end-to-end differentiability of such simulators allows us to efficiently optimize their parameters, which means the physics engine can be incorporated into learning-based architectures. By augmenting the original physics engine with neural networks, even unmodeled effects can be captured by such learning-based simulator (see [NeuralSim](https://eric-heiden.com/publication/2021-neuralsim-icra/)).
