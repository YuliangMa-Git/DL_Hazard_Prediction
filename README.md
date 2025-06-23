# Deep Learning-based Proactive Hazard Prediction for Human-Robot Collaboration with Sensor Malfunctions
Yuliang Ma*, Zilin Jin, Qi Liu, Ilshat Mamaev, and Andrey Morozov

{yuliang.ma@ias.uni-stuttgart.de}

This work has been accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025).

<img src="/source/Hazard_demo.png" height="220" />
<img src="/source/framework.png" height="360" />

## Abstract
Safety is a critical concern in human-robot collaboration (HRC). As collaborative robots take on increasingly complex tasks in human environments, their systems have become more sophisticated through the integration of multimodal sensors, including force-torque sensors, cameras, LiDARs, and IMUs. However, existing studies on HRC safety primarily focus on ensuring safety under normal operating conditions, overlooking scenarios where internal sensor faults occur. 

While anomaly detection modules can help identify sensor errors and mitigate hazards, two key challenges remain: (1) no anomaly detector is flawless, and (2) not all sensor malfunctions directly threaten human safety. Relying solely on anomaly detection can lead to missed errors or excessive false alarms.

To enhance safety in real-world HRC applications, this paper introduces a deep learning-based method that proactively predicts hazards following the detection of sensory anomalies. We simulate two common types of faults—bias and noise—affecting joint sensors and monitor abnormal manipulator behaviors that could pose risks in fenceless HRC environments. A dataset of 2,400 real-world samples is collected to train the proposed hazard prediction model.

The approach leverages multimodal inputs, including RGB-D images, human pose, joint states, and planned robot paths, to assess whether sensor malfunctions could lead to hazardous events. Experimental results show that the proposed method outperforms state-of-the-art models, while offering faster inference speed. Additionally, cross-scenario testing confirms its strong generalization capabilities.
## Video
[\[video\]](https://youtu.be/wVkbuf_aoXI) 
## Dataset
[\[dataset\]](https://www.kaggle.com/datasets/yuliangma/dl-hazard-prediction) 
## Citation
