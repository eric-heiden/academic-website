---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "DiSECt: A Differentiable Simulation Engine for Autonomous Robotic Cutting"
authors:
- "Eric Heiden"
- "Miles Macklin"
- "Yashraj Narang"
- "Dieter Fox"
- "Animesh Garg"
- "Fabio Ramos"
date: 2021-06-03
doi: "10.15607/RSS.2021.XVII.067"

# Schedule page publish date (NOT publication's date).
publishDate: 2021-06-03T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "Robotics: Science and Systems (RSS) 2021"
publication_short: "We introduce a differentiable simulator for robotic cutting. It achieves highly accurate predictions of the knife forces, optimizes cutting actions & more! <span style='color:#f60'>Best Student Paper at RSS 2021</span>"

abstract: "Robotic cutting of soft materials is critical for applications such as food processing, household automation, and surgical manipulation. As in other areas of robotics, simulators can facilitate controller verification, policy learning, and dataset generation. Moreover, <em>differentiable</em> simulators can enable gradient-based optimization, which is invaluable for calibrating simulation parameters and optimizing controllers. In this work, we present DiSECt - the first differentiable simulator for cutting soft materials. The simulator augments the finite element method (FEM) with a continuous contact model based on signed distance fields (SDF), as well as a continuous damage model that inserts springs on opposite sides of the cutting plane and allows them to weaken until zero stiffness to model crack formation. Through various experiments, we evaluate the performance of the simulator. We first show that the simulator can be calibrated to match resultant forces and deformation fields from a state-of-the-art commercial solver and real-world cutting datasets, with generality across cutting velocities and object instances. We then show that Bayesian inference can be performed efficiently by leveraging the differentiability of the simulator, estimating posteriors over hundreds of parameters in a fraction of the time of black-box methods. Finally, we illustrate that control parameters in the simulation can be optimized to minimize cutting forces via lateral slicing motions."

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

links:
- name: Website
  url: https://diff-cutting-sim.github.io
- name: Blog post
  url: https://developer.nvidia.com/blog/nvidia-research-disect-a-differentiable-simulation-engine-for-autonomous-robotic-cutting/

url_pdf: https://arxiv.org/abs/2105.12244
url_code: 
url_dataset:
url_poster: poster.pdf
url_project:
url_slides:
url_source:
url_video: https://youtu.be/zUjdNR6kpfI

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

## Presentation at RSS 2021

<iframe width="720" height="405" src="https://www.youtube.com/embed/zUjdNR6kpfI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very honored that DiSECt won a Best Student Paper Award at <a href="https://twitter.com/hashtag/RSS2021?src=hash&amp;ref_src=twsrc%5Etfw">#RSS2021</a>! Congrats to my co-authors <a href="https://twitter.com/milesmacklin?ref_src=twsrc%5Etfw">@milesmacklin</a>, Yashraj Narang, Dieter Fox, <a href="https://twitter.com/animesh_garg?ref_src=twsrc%5Etfw">@animesh_garg</a> and Fabio Ramos, and thanks for this great collaboration <a href="https://twitter.com/NVIDIAAI?ref_src=twsrc%5Etfw">@NVIDIAAI</a>! 🎉 <a href="https://t.co/An2RYMXIl2">https://t.co/An2RYMXIl2</a></p>&mdash; Eric Heiden (@eric_heiden) <a href="https://twitter.com/eric_heiden/status/1416054055923822594?ref_src=twsrc%5Etfw">July 16, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 
