---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Probabilistic Inference of Simulation Parameters via Parallel Differentiable Simulation"
authors:
- "Eric Heiden"
- "Christopher E. Denniston"
- "David Millard"
- "Fabio Ramos"
- "Gaurav S. Sukhatme"
date: 2021-09-20
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2021-09-20T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "International Conference on Robotics and Automation (ICRA) 2021"
publication_short: ""

abstract: " To accurately reproduce measurements from the real world, simulators need to have an adequate model of the physical system and require the parameters of the model be identified.

We address the latter problem of estimating parameters through a Bayesian inference approach that approximates a posterior distribution over simulation parameters given real sensor measurements. By extending the commonly used Gaussian likelihood model for trajectories via the multiple-shooting formulation, our chosen particle-based inference algorithm Stein Variational Gradient Descent is able to identify highly nonlinear, underactuated systems. We leverage GPU code generation and differentiable simulation to evaluate the likelihood and its gradient for many particles in parallel.

Our algorithm infers non-parametric distributions over simulation parameters more accurately than comparable baselines and handles constraints over parameters efficiently through gradient-based optimization. We evaluate estimation performance on several physical experiments. On an underactuated mechanism where a 7-DOF robot arm excites an object with an unknown mass configuration, we demonstrate how our inference technique can identify symmetries between the parameters and provide highly accurate predictions."

# Summary. An optional shortened abstract.
summary: "Reducing the reality gap through a Bayesian inference algorithm that leverages massive GPU parallelism and differentiable simulators."

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
  url: https://uscresl.github.io/prob-diff-sim/

url_pdf: https://arxiv.org/abs/2109.08815
url_code:
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video: https://youtu.be/QfhsWlKbfYM

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "We reduce the reality gap in robotics simulators by introducing a Bayesian inference approach named Constrained Stein Variational Gradient Descent (CSVGD)"
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
