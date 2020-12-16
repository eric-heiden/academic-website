---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Scaling simulation-to-real transfer by learning composable robot skills"
authors: ["Ryan C. Julian", "Eric Heiden", "Zhanpeng He", "Hejia Zhang", "Stefan Schaal", "Joseph J. Lim", "Gaurav S. Sukhatme", "Karol Hausman"]
date: 2019-10-29T20:15:43-07:00
doi: "10.1177/0278364920944474"

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T20:15:43-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "International Journal of Robotics Research"
publication_short: ""

abstract: "We present a strategy for simulation-to-real transfer, which builds on recent advances in robot skill decomposition. 
Rather than focusing on minimizing the simulation-reality gap, we learn a diverse set of skills and their variations, and embed those skill variations in a single continuously-parameterized space.
We then interpolate, search, and plan in this space to find a transferable policy which solves more complex, high-level tasks by combining low-level skills and their variations.
We first characterize the behavior of this learned skill space, by experimenting with several techniques for composing pre-learned latent skills.
We then discuss an algorithm which allows our method to perform long-horizon tasks never seen in simulation, by intelligently sequencing short-horizon latent skills. Our algorithm adapts to unseen tasks online by repeatedly choosing new skills from the latent space, using live sensor data and simulation to predict which latent skill will perform best next in the real world.
Importantly, our method learns to control a real robot in joint-space to achieve these high-level tasks with little or no on-robot time, despite the fact that the low-level policies may not be perfectly transferable from simulation to real, and that the low-level skills were not trained on any examples of high-level tasks.
In addition to our results indicating a lower sample complexity for families of tasks, we believe that our method provides a promising template for combining learning-based methods  with proven classical robotics algorithms such as model-predictive control (MPC)."

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

url_pdf: https://journals.sagepub.com/doi/abs/10.1177/0278364920944474
url_code: https://github.com/ryanjulian/embed2learn
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video: https://youtu.be/Syr2RQTHqTs

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
projects: ["tamp"]

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
