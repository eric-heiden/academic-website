---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Augmenting Differentiable Simulators with Neural Networks to Close the Sim2Real Gap"
authors:
- "Eric Heiden"
- "David Millard"
- "Erwin Coumans"
- "Gaurav S. Sukhatme"

date: 2020-07-12
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2020-07-12T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["4"]

# Publication name and optional abbreviated publication name.
publication: "R:SS 2020 Workshop on Closing the Reality Gap in Sim2Real Transfer for Robotics"
publication_short: ""

abstract: "We present a differentiable simulation architecture for articulated rigid-body dynamics that enables the augmentation of analytical models with neural networks at any point of the computation. Through gradient-based optimization, identification of the simulation parameters and network weights is performed efficiently in preliminary experiments on a real-world dataset and in sim2sim transfer applications, while poor local optima are overcome through a random search approach."

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

url_pdf: https://arxiv.org/abs/2007.06045
url_code: https://github.com/google-research/tiny-differentiable-simulator
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video: https://www.youtube.com/watch?v=awhkI5xtFa0

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Force-based contact model augmented by a neural network"
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
<video controls loop autoplay>
  <source src="neural_contact.mp4" type="video/mp4">
</video>