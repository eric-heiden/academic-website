---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Physics-based Simulation of Continuous-Wave LIDAR for Localization, Calibration and Tracking"
authors: ["Eric Heiden", "Ziang Liu", "Ragesh K. Ramachandran", "Gaurav S. Sukhatme"]
date: 2019-10-29
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2019-10-29T15:33:15-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "Under review at International Conference on Robotics and Automation (ICRA) 2020"
publication_short: ""

abstract: "Light Detection and Ranging (LIDAR) sensors play an important role in the perception stack of autonomous robots, supplying mapping and localization pipelines with depth measurements of the environment.
While their accuracy outperforms other types of depth sensors, such as stereo or time-of-flight cameras, the accurate modeling of LIDAR sensors requires laborious manual calibration that typically does not take into account the interaction of laser light with different surface types, incidence angles and other phenomena that significantly influence measurements. In this work, we introduce a physically plausible model of a 2D continuous-wave LIDAR that accounts for the surface-light interactions and simulates the measurement process in the Hokuyo URG-04LX LIDAR. Through automatic differentiation, we employ gradient-based optimization to estimate model parameters from real sensor measurements."

# Summary. An optional shortened abstract.
summary: ""

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

url_pdf: https://arxiv.org/abs/1912.01652
url_code:
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video: https://youtu.be/hGphZbMDbYA

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Illustration of the measurement process of a continuous-wave laser scanner."
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
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">So what did the car actually see?<br>Well, it turns out at first it looked like an obstacle so it slowed down some, then it decided that nah, there&#39;s nothing and lifted it&#39;s &quot;foot&quot; off the brakes, huh? <a href="https://t.co/bun4mrxKOZ">pic.twitter.com/bun4mrxKOZ</a></p>&mdash; green (@greentheonly) <a href="https://twitter.com/greentheonly/status/1202782357046079489?ref_src=twsrc%5Etfw">December 6, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 

<blockquote class="twitter-tweet" data-conversation="none"><p lang="en" dir="ltr">Now we wondered what would happen if we remove the foil? Well, pretty much the same thing only this time autopilot actually sped up before impact after initially slowing down some and did not disengage until I stomped on the brakes. <a href="https://t.co/NgszJwO2zh">pic.twitter.com/NgszJwO2zh</a></p>&mdash; green (@greentheonly) <a href="https://twitter.com/greentheonly/status/1202782421005017093?ref_src=twsrc%5Etfw">December 6, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 
