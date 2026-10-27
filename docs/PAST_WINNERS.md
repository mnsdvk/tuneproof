# What past OpenCV competition winners tell us

Sources: opencv.org competition pages and announcements, I-Programmer coverage (links in chat reply). Facts are from those pages; **"Inference" lines are my interpretation, not stated by OpenCV or the judges.**

## Facts found
| Edition | Scale | Winners / finalists seen | Notes |
|---|---|---|---|
| 2020 Spatial AI (Intel) | 235 proposals; 32 Phase-1 winners (+22 strong entries) | 6 final winners sharing $12,000 (originally planned 3 prizes / $6k). Phase-1 examples: *Cyclops* (spatial AI + NLP assistant for visually impaired people), agricultural semantic segmentation, robot arm picking glass bottles, indoor-landmark graph SLAM | Proposal first, then hardware-backed build. I-Programmer noted final projects were *more ambitious* than their proposals after teams got hands-on time. |
| 2021 AI Competition (Intel + Azure) | 1,400+ proposals; 210–257 finalist teams; 1,200 OAK-D devices | Global: **#1 Cortic Tigers** (democratizing AI on OAK-D), **#2 Caleta** (NeoCam: real-time telemonitoring of preterm neonates), **#3 Cerebros** (next-gen electric wheelchair). Regional examples: American Stereo Types (VR laser tag), NaRiPa VisionAIrs (6D-pose grasping for retail robotics), Train & Track (inspection of train current collectors) | Region-based prizes + popular vote. |
| 2022 Spatial AI Contest | 50 finalist teams, LEGO-based replicas of industrial manufacturing | **1st ScanNERF** (team EyeCan Reborn; Neural Radiance Field capture), **2nd UCA Robotics** (vision-based human–robot interaction), **3rd Potato Lovers** (Colorado beetle pest control); featured also: AI-powered Petri-dish scanner; Popular vote: Barolo / ARNEIS | Public webinar + Microsoft Build showcase for winners. |
| 2026 (this one) | $20,250 prizes; ~55 × $150 | Rubric: Technical execution 30, Innovation 20, Impact 20, UX 10, Docs/presentation 10, Cloud/repro/responsible 10 | Judges named: Phil Nelson, Gary Bradski. |

## Patterns (descriptive)
1. **Real-world, human-centred problems dominate**: healthcare (neonates), accessibility (visually impaired, wheelchair), agriculture, industrial inspection.
2. **A new capability was showcased, not just a model**: spatial/depth, NeRF capture, human–robot interaction, 6D pose. Winners combined perception with action in a physical or interactive loop.
3. **Working demos with video** were central (winners were interviewed/demoed in webinars).
4. **Winning entries were more ambitious than their proposals** (2020 coverage) — iterating hands-on mattered.
5. **Sponsor-technology depth is rewarded**: 2020/2021 built around OAK-D, 2022 around OAK-D-Lite; 2026 around **OpenCV 5 + AWS Graviton COOL**.

## Inference for this 2026 entry (not stated by organizers)
- Judges likely reward *measured* evidence tied to the sponsor stack: here that means a reproducible COOL-vs-pip benchmark on Graviton and a trace proving visual evidence changes decisions. VisionPilot is built to hand over exactly those artefacts.
- The rubric gives 30% to technical depth of OpenCV 5 use and 10% to cloud reproducibility — a clean repo, pinned dependencies, one-command evaluation, and honest limitations score directly.
- A **human-in-the-loop safety/QA story** matches both the "responsible operation" criterion and the proven appeal of healthcare/safety themes.
- No one can guarantee a win: 55 prizes exist, but the top slots depend on competing entries we cannot see. The best lever is polish of the 5-minute video and measured results.
