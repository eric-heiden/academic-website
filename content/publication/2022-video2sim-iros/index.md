---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Inferring Articulated Rigid Body Dynamics from RGBD Video"
authors:
- "Eric Heiden"
- "Ziang Liu"
- "Vibhav Vineet"
- "Erwin Coumans"
- "Gaurav S. Sukhatme"
date: 2022-03-21
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2022-03-20T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "To appear at International Conference on Intelligent Robots and Systems (IROS) 2022"
publication_short: ""

abstract: "Being able to reproduce physical phenomena ranging from light interaction to contact mechanics, simulators are becoming increasingly useful in more and more application domains where real-world interaction or labeled data are difficult to obtain. Despite recent progress, significant human effort is needed to configure simulators to accurately reproduce real-world behavior. We introduce a pipeline that combines inverse rendering with differentiable simulation to create digital twins of real-world articulated mechanisms from depth or RGB videos. Our approach automatically discovers joint types and estimates their kinematic parameters, while the dynamic properties of the overall mechanism are tuned to attain physically accurate simulations. Control policies optimized in our derived simulation transfer successfully back to the original system, as we demonstrate on a simulated system. Further, our approach accurately reconstructs the kinematic tree of an articulated mechanism being manipulated by a robot, and highly nonlinear dynamics of a real-world coupled pendulum mechanism."

# Summary. An optional shortened abstract.
summary: 'We present a pipeline to learn simulators from depth or RGB video. The "URDF" of a mechanism is reconstructed, and the simulation parameters are inferred through Bayesian inference.'

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
  url: https://eric-heiden.github.io/video2sim/

url_pdf: https://arxiv.org/abs/2203.10488
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
  caption: "Simulation inferred from a video of Rott's coupled pendulum mechanism"
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
