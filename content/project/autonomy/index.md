---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Autonomous navigation"
summary: "Navigate unknown environments while coping with noisy sensor measurements."
authors: []
tags: []
categories: []
date: 2019-10-29T17:43:35-07:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "&copy; [CoSTAR](https://subt.jpl.nasa.gov/)"
  focal_point: ""
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

I have worked on autonomous robot navigation involving quadrotors that build a map of the environment while coping with perception uncertainty. In my joint work with Ali-akbar Agha-mohammadi on [Confidence-rich grid Mapping (CRM)](/publication/2019-crm-ijrr/), we extended the commonly used voxel-based robot mapping algorithm to account for the interdependence of voxels that are updated from the same depth measurements, leading to more accurate uncertainty predictions compared to previous works. Such map representation gives rise to new trajectory generation algorithms that account for the perception uncertainty.

To this end, in [Planning High-speed Safe Trajectories in Confidence-rich Maps](/publication/2017-crm-to-iros) we propose a trajectory optimization algorithm that maximizes the Lower Confidence Bound that trades off the expected value of reachability of the current solution candidate versus the estimated uncertainty of such reachability value. The resulting trajectories exhibit motions where the robot is preemptively moving itself toward areas of the environment from which it has a steady perception of the upcoming segment of the path to take.

In collaboration with JPL, where I spent the summer 2018 as an intern under the guidance of Ali-akbar Agha-mohammadi, I extended the mapping solution to fuse measurements from multiple robots equipped with heterogeneous sensors, such as stereo cameras and lidars, in [Heterogeneous Sensor Fusion via Confidence-rich 3D Grid Mapping: Application to Physical Robots](/publication/2018-crm-fusion-iser). The map updates are combined efficiently in real time and their respective probabilistic measurement models determine the noise characteristics through which CRM calculates the voxel updates. In our experiments, we demonstrated the mapping framework working in tandem on the Intel Aero drone equipped with a RealSense depth sensor and a downward-facing 2D LIDAR, plus a novel wheeled robot that is equipped with two planar LIDAR sensors that spin as the wheels rotate.

In ongoing work with the NASA Jet Propulsion Laboratory (JPL) I am contributing to the perception stack of team [CoSTAR](https://subt.jpl.nasa.gov/) (JPL/Caltech/MIT/KAIST/LTU) in the [DARPA SubT Challenge](https://www.subtchallenge.com/) where ensembles of heterogeneous robots are tasked to explore and map underground tunnels and mines under degraded visual conditions.
