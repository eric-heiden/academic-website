---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Closing the Sim2Real Gap using Invertible Simulators"

authors:
- "Eric Heiden"

date: 2020-07-11T20:15:43-07:00
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: 2020-07-11T20:15:43-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["4"]

# Publication name and optional abbreviated publication name.
publication: "[RSS Pioneers 2020](https://sites.google.com/view/rsspioneers2020/home) Workshop"
publication_short: ""

abstract: ""

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
url_project: https://sites.google.com/view/rsspioneers2020/home
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
projects: ["ids"]

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---

Employing robots in the real world to perform a large variety of tasks remains a great challenge to current perception, planning and control algorithms. Various specialized representations, such as for mapping or localization, have been proposed which are typically used in fixed pipelines that fuse perception, planning and control. These approaches are typically highly interpretable in a way that humans can reason about the prediction uncertainty of the system, or what additional measurements are necessary to improve the predictions. On the other hand, such static frameworks do not allow the robot to learn from experience or adapt to changing task requirements.

Learning-based approaches have found great success in domains where large amounts of labelled data is available. Many problems in robotics, however, do not belong to such regime where training data is easily obtainable. Instead, it is often only possible to provide few kinesthetic demonstrations, or rely entirely on self-supervised or reinforcement learning. While the longstanding motivation behind such learning approaches is to enable robots to improve by learning from their own experience, the current instantiations of state-of-the-art reinforcement learning (RL) algorithms, even model-based, require extensive amounts of interaction samples, such that, in most cases, simulators are necessary to provide a risk-free environment that runs orders of magnitude faster than real time. With regards to accountable AI, many of these approaches are not human-interpretable -- they may achieve high performance on certain tasks but it remains an open research question how the training setup must be designed to guarantee performance throughout all metrics over the tasks of interest.

Whether control policies are learned through reinforcement learning, or feedback control laws are optimized -- in most approaches simulators are used to validate the algorithms before deploying them on the real system. An inherent problem to such techniques is the disparity between the simulated and the real world, i.e., the sim2real gap. Various methods have been proposed to overcome this issue, such as domain randomization and domain adaption.

In this work, we approach the problem of visuomotor control from a different angle. Instead of learning a separate model or policy in a simulator, we use the simulator as the model that we can use to derive controllers and estimate the state of our system of interest. Simulators, such as physics engines, already encode our understanding of the world through the laws of physics and generalize well to a wide variety of application scenarios. Most quantities, such as the geometry of the objects, are human-interpretable and can be even verified through a variety of specialized tools. We propose to design the simulator from the ground up to be *invertible*, i.e., such that we can estimate the simulation settings from the observations of the real system.
