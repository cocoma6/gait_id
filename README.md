# Gait silhouette generation and gait recognition

## Datasets USed
+ KTH: https://www.csc.kth.se/cvap/actions/ <br>
+ Kaggle version: https://www.kaggle.com/datasets/saimadhurivasam/human-activity-recognition-from-video <br>
+ TUM-Gait: https://www.ce.cit.tum.de/en/mmk/verschiedenes/tum-iitkgp-gait-database/

## TODO
1. Gait silhouette generation from videos of a SINGLE HUMAN with NO noise
2. Gait silhouette generation from videos of a MULTIPLE HUMAN with NO noise
3. Gait recognition TBA

## Notes
gait-pretreat.py: modify OUTPUT_PATH, POST_PATH, INPUT_PATH

gait-train.py: modify the config part

The model repo is based on [GaitSet](https://github.com/AbnerHqC/GaitSet) and the related paper is **[GaitSet: gait recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351667)**. There are some other papers which may be helpful: [Gait Global-Local Feature Representation](https://arxiv.org/pdf/2208.01380.pdf), [Gait signature involves Optical Flow](https://arxiv.org/pdf/1603.01006.pdf).