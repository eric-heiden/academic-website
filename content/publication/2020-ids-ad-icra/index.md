---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Automatic Differentiation and Continuous Sensitivity Analysis of Rigid Body Dynamics"
authors: ["David Millard*", "Eric Heiden*", "Shubham Agrawal", "Gaurav S. Sukhatme"]
date: 2019-10-29T10:51:34-07:00
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T19:51:34-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["4"]

# Publication name and optional abbreviated publication name.
publication: ""
publication_short: ""

abstract: "A key ingredient to achieving intelligent behavior is physical understanding that equips robots with the ability to reason about the effects of their actions in a dynamic environment. Several methods have been proposed to learn dynamics models from data that inform model-based control algorithms. While such learning-based approaches can model locally observed behaviors, they fail to generalize to more complex dynamics and under long time horizons.
In this work, we introduce a differentiable physics simulator for rigid body dynamics. Leveraging various techniques for differential equation integration and gradient calculation, we compare different methods for parameter estimation that allow us to infer the simulation parameters that are relevant to estimation and control of physical systems. In the context of trajectory optimization, we introduce a closed-loop model-predictive control algorithm that infers the simulation parameters through experience while achieving cost-minimizing performance."

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

url_pdf: https://arxiv.org/abs/2001.08539
url_code:
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video:

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Simulated cartpole systems in MuJoCo (left column) and our physics engine (right column)."
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
