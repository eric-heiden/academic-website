---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Confidence-rich Grid Mapping"
authors: ["Ali-akbar Agha-mohammadi", "Eric Heiden", "Karol Hausman", "Gaurav S. Sukhatme"]
date: 2019-05-01
doi: "10.1177/0278364919839762"

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T20:01:52-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "International Journal of Robotics Research"
publication_short: ""

abstract: "Representing the environment is a fundamental task in enabling robots to act autonomously in unknown environments. In this work, we present confidence-rich mapping (CRM), a new algorithm for spatial grid-based mapping of the 3D environment. CRM augments the occupancy level at each voxel by its confidence value. By explicitly storing and evolving confidence values using the CRM filter, CRM extends traditional grid mapping in three ways: first, it partially maintains the probabilistic dependence among voxels; second, it relaxes the need for hand-engineering an inverse sensor model and proposes the concept of sensor cause model that can be derived in a principled manner from the forward sensor model; third, and most importantly, it provides consistent confidence values over the occupancy estimation that can be reliably used in collision risk evaluation and motion planning. CRM runs online and enables mapping environments where voxels might be partially occupied. We demonstrate the performance of the method on various datasets and environments in simulation and on physical systems. We show in real-world experiments that, in addition to achieving maps that are more accurate than traditional methods, the proposed filtering scheme demonstrates a much higher level of consistency between its error and the reported confidence, hence, enabling a more reliable collision risk evaluation for motion planning."

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

url_pdf: "https://journals.sagepub.com/doi/abs/10.1177/0278364919839762"
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
  caption: "Occupancy grids computed via the log-odds approach from OctoMap (top row) and our proposed CRM (bottom row) resulting from the ICL-NUIM living room dataset."
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
