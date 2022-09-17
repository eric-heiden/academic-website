---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Grasp'D: Differentiable Contact-rich Grasp Synthesis for Multi-fingered Hands"
authors:
- "Dylan Turpin"
- "Liquan Wang"
- "Eric Heiden"
- "Yun-Chun Chen"
- "Miles Macklin"
- "Stavros Tsogkas"
- "Scen Dickinson"
- "Animesh Garg"
date: 2022-08-21
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2022-08-20T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "European Conference on Computer Vision (ECCV) 2022"
publication_short: ""

abstract: "The study of hand-object interaction requires generating viable grasp poses for high-dimensional multi-finger models, often relying on analytic grasp synthesis which tends to produce brittle and unnatural results. This paper presents Grasp’D, an approach for grasp synthesis with a differentiable contact simulation from both known models as well as visual inputs. We use gradient-based methods as an alternative to sampling-based grasp synthesis, which fails without simplifying assumptions, such as pre-specified contact locations and eigengrasps. Such assumptions limit grasp discovery and, in particular, exclude high-contact power grasps. In contrast, our simulation-based approach allows for stable, efficient, physically realistic, high-contact grasp synthesis, even for gripper morphologies with high-degrees of freedom. We identify and address challenges in making grasp simulation amenable to gradient-based optimization, such as non-smooth object surface geometry, contact sparsity, and a rugged optimization landscape. Grasp’D compares favorably to analytic grasp synthesis on human and robotic hand models, and resultant grasps achieve over 4× denser contact, leading to significantly higher grasp stability."

# Summary. An optional shortened abstract.
summary: 'Synthesize contact-rich grasps for multi-fingered hands via differentiable simulation'

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
  url: https://graspd-eccv22.github.io/

url_pdf: https://arxiv.org/abs/2208.12250
url_code:
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video: https://youtu.be/tkl_lywZ94o

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Example grasps generated for the MANO hand"
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: ["ids", "tamp"]

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
