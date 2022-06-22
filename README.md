# RVS-VoteNet
An implementation of the paper: RVS-VoteNet: Revisiting VoteNet with Inner-group Relation and Weighted Relation-Aware Proposal

In this work, we revisit and dive
deeper into VoteNet, one of the most influential yet underfully-explored 3D object detection networks, and develop a
more accurate 3D detection model in cluttered indoor scenes.
Specifically, we rethink point cloud feature extraction, Hough
voting and proposal generation in VoteNet, finding that each
step has some inherent defects. In feature extraction, we design
Inner-Group Relation module, an effective alternative to the
vanilla set-abstraction module, which can fully integrate the
local geometry/semantic information and global shape/semantic
information into point feature embedding. Equipped with a
lightweight feature affline module and channel-wise attention
machanism, our Inner-Group Relation module can easily learn
better point representation from each local neighbour point
cloud. In hough voting, we propose a newly non-voting loss to
suppress the effect of the background votes, further boosting
the detection performance. During object proposal generation,
we utilize the contextual information to tell the difference when
the instance geometry is incomplete or featureless. However,
adopting relations between all the object proposals for detection
is inefficient, so we propose a weighted Relation-aware Proposal
Module that uses an objectness-aware manner to weigh the
relation importance for a better proposal generation. Equipped
with such three improvements, the proposed method achieves
state-of-the-art 3D object detection performance on two widely
used benchmarks, ScanNet V2 and SUN RGB-D.
