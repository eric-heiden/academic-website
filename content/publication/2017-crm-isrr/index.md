---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Confidence-rich Grid Mapping"
authors: ["Ali-akbar Agha-mohammadi", "Eric Heiden", "Karol Hausman", "Gaurav S. Sukhatme"]
date: 2017-11-01
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T17:57:19-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "International Symposium on Robotics Research (ISRR)"
publication_short: ""

abstract: "Occupancy grids are a common framework in robotics for creating a spatial map of the environment. 
Traditional grid mapping algorithms assume that map voxel occupancies are independent of each other. 
In addition, they use a map representation where each voxel stores a single number representing the occupancy probability. 
This leads to inconsistencies in the map and, more importantly, conflicts between the map error and the reported confidence values, resulting in critical cases of overconfidence.
Such discrepancies pose challenges for planners that rely on the generated map for collision avoidance.
This paper studies occupancy grids from a planning perspective and proposes a novel algorithm for grid mapping in the presence of noisy measurements.
By storing richer data at each voxel, the new grid representation enables an accurate estimate of the variance of occupancy.
We show that, in addition to achieving maps that are more accurate than traditional methods, the proposed filtering scheme demonstrates a much higher level of consistency between its error and the reported confidence."

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
url_slides:
url_source:
url_video:

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
projects: ["autonomy"]

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
