# Introduction
In this project, our primary objective is to conduct a comprehensive research study to determine the most suitable state-of-the-art pre-trained COCO model for the precise labeling of ground-based objects within an aerial object dataset. This research is crucial to enhance the accuracy and efficiency of our object detection process in an aerial and ground-based context.

# DINO model
The DETR (DEtection TRansformer) with Improved DeNoising Anchor Boxes (DINO) is an innovative research project that enhances end-to-end object detection by introducing a novel approach to model architecture and Contrastive DeNoising Training (CDT) [1, 2]. The key features of this model are:

* Model architecture: The heart of the DINO approach lies in its adaptation of the DETR architecture, which has already shown remarkable promise in the field of object detection. DETR replaces the traditional anchor boxes and non-maximum suppression mechanisms found in CNN with a Transformer-based architecture. However, DINO takes DETR a step further by improving the way anchor boxes are used. In DINO, the model employs a novel mechanism for generating anchor boxes that are adapted to the specific characteristics of the dataset. Unlike conventional anchor boxes that are predefined and fixed, DINO’s anchor boxes are dynamically adjusted based on the features learned by the model. This adaptability allows the model to better align the anchor boxes with the objects in the image, leading to improved localization accuracy.

* One of the ground-breaking aspects of DINO is its use of CDT to enhance the robustness and generalization capabilities of the model. CDT is a self-supervised learning technique that leverages the power of contrastive learning to train the model.
In the context of DINO, CDT is applied during the pre-training phase. The model ispresented with pairs of noisy images, where the noise can be in the form of various perturbations, occlusions, or distortions. The objective is to make the model learn to distinguish between anchor boxes and object features in the presence of these perturbations.This training forces the model to focus on the most discriminative features and helps it become more resilient to noise and clutter in real-world images.

* End-to-End Object Detection: DINO’s end-to-end object detection approach offers a seamless and unified framework for both localization and classification tasks. The model processes an entire image as a sequence of patches, leveraging the self-attention mechanism within the Transformer architecture to capture global and local context simultaneously. The dynamic anchor boxes, learned during training, are used to predict object locations.

By integrating the CDT into this end-to-end architecture, DINO achieves state-of-the-art results in object detection tasks. Its ability to adapt anchor boxes and robustly handle noisy inputs allows it to excel in scenarios where conventional object detectors might struggle.

![DINO Architecture](img/DINO_architecture.png)

# References
[1] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. “DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection”. In: (Mar. 2022). URL: https://arxiv.org/abs/2203.03605v4.

[2] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. GitHub - IDEA-Research/DINO: [ICLR 2023] Official implementation of the paper ”DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection”. URL: https://github.com/IDEA-Research/DINO.
