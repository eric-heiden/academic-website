---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Planning High-speed Safe Trajectories in Confidence-rich Maps"
authors: ["Eric Heiden", "Karol Hausman", "Gaurav S. Sukhatme", "Ali-akbar Agha-mohammadi"]
date: 2017-11-01
doi: "10.1109/IROS.2017.8206120"

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T19:34:44-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)"
publication_short: ""

abstract: "Planning safe, high-speed trajectories in unknown environments remains a major roadblock on the way toward achieving fast autonomous flight.
Current state-of-the-art planning approaches use sampling-based methods or trajectory optimization to obtain fast trajectories, whose safety is evaluated by taking into account the current state estimate of the environment.
In unknown environments, however, this leads to numerous stops caused by the need for re-planning the trajectory due to unexpected obstacles. 
In this paper, we propose to use an active perception paradigm for planning. We predict the future uncertainty of the map and optimize trajectories to minimize re-planning risk. This leads to faster _and_ safer trajectories.
We evaluate the proposed planning approach in a series of simulation experiments, which show that we are able to achieve safer trajectories with a smaller number of re-planning stops and faster speeds."

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
url_code:
url_dataset:
url_poster:
url_project:
url_slides: 2017-planning-slides.pdf
url_source:
url_video:

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Resulting trajectories with our proposed Lower Confidence Bound (LCB) optimization objective."
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: ["autonomy", "tamp"]

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
