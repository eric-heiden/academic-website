---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Gradient-Informed Path Smoothing for Wheeled Mobile Robots"
authors: ["Eric Heiden", "Luigi Palmieri", "Sven Koenig", "Kai O. Arras", "Gaurav S. Sukhatme"]
date: 2018-05-01
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T18:06:58-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "In International Conference on Robotics and Automation (ICRA)"
publication_short: ""

abstract: "Planning smooth trajectories is important for the safe, efficient and
comfortable operation of mobile robots, such as wheeled robots moving in
crowded environments or cars moving at high speed. Asymptotically optimal
sampling-based motion planners can be used to generate such trajectories.
However, to achieve the necessary efficiency for the real-time
operation of robots, one often uses their initial feasible trajectories or the
trajectories of non-optimal motion planners instead, typically after a post-smoothing
step. We propose a gradient-informed post-smoothing algorithm, called GRIPS,
that deforms given trajectories by locally optimizing the placement of
vertices while satisfying the system's kinodynamic constraints.  We show
experimentally that GRIPS typically produces trajectories of significantly smaller length 
and higher smoothness than several existing post-smoothing
algorithms."

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
url_code: "https://github.com/eric-heiden/grips"
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video: "https://youtu.be/j7knKdN8Hg4"

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "An example path generated by Theta∗ with Reeds-Shepp steering using Gradient-Informed Path Smoothing (GRIPS) (solid line)."
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