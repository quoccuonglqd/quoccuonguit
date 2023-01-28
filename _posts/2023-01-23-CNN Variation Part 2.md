---
title:  "Learning Note On Convotional Neural Network Variants Part 2"
permalink: /blogs/Convolutional Neural Network Variants Part 2
excerpt: ""
header:
  # overlay_image: /assets/images/maxresdefault.jpg
  teaser: ../assets/images/CNN/cnnteaser.jpg
  caption: ""
  actions:
    - label: "More Info"
      url: "https://www.matthewtancik.com/nerf"
categories:
  - Deep Learning
toc: true
related: true
---

# 1/ Introduction:

Hello and welcome to my blog post on famous variants of convolutional neural networks (CNNs)! If you're reading this, chances are you're just as passionate about deep learning and computer vision as I am. CNNs are an integral part of the field, and have been responsible for some of the most impressive breakthroughs in image classification and object recognition. In this post, I'll be introducing you to some of the most famous CNNs that have been developed over the years and discussing their unique characteristics and contributions to the field. Whether you're a seasoned deep learning practitioner or just starting out, I hope this post will inspire you to dive deeper into the world of CNNs and learn more about how they work. So without further ado, let's get started!

# 2/ Famous network:

## 2.7/ EfficientNet:

EfficientNet is a convolutional neural network (CNN) architecture that has been designed to improve upon previous state-of-the-art models by increasing the model's capacity while also reducing the number of parameters and computational cost. The EfficientNet architecture is achieved through a combination of techniques such as compound scaling, which adjusts the resolution, depth, and width of the network in a systematic and principled manner, and the use of a mobile inverted bottleneck (MBConv) block, which is a more efficient version of the standard inverted bottleneck block.

The study found that a good balance between the width, depth, and resolution of the network is important for the model's performance. And that this balance can be achieved by proportionally adjusting the width, depth, and resolution with the same scaling factor. The author also provided an example of how to increase the computational resources used by the network by a factor of $2^N$. The method proposed in this statement is to increase the network depth, width, and image size simultaneously with a scaling factor that is determined by a small grid search on the original small model. The idea is to use three different constant coefficients, $\alpha$, $\beta$, and $\gamma$, such that $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2\alpha$ $\alpha \geq 1, \beta \geq 1, \gamma \geq 1$ to scale the depth, width, and image size respectively. These coefficients are determined by a small grid search on the original small model, which means that a small range of values for each coefficient is tested, and the best coefficients are selected based on the model's performance. For example, if we want to use 8 times more computational resources ($N=3$), then we can increase the network depth by $α^3$, width by $β^3$ and image size by $γ^3$. This will increase the computational resources used by the network by a factor of $2^3 = 8$.

The authors developed their baseline network by using a multi-objective neural architecture search (NAS) that aims to optimize both accuracy and FLOPS (floating point operations per second). The main building block is mobile inverted bottleneck from MobileNetV2, combined with squeeze-and-excitation optimization.

The EfficientNetV2 is similar to the original EfficientNet, but it uses a upgraded neural architecture search (NAS) to find the baseline architecture. The NAS framework used in EfficientNetV2 is based on previous NAS works and aims to jointly optimize accuracy, parameter efficiency, and training efficiency on modern accelerators. The search space used in EfficientNetV2 is a stage-based factorized space that consists of design choices for convolutional operation types, number of layers, kernel size, and expansion ratio. The search space is also reduced by removing unnecessary search options and reusing the same channel sizes from the backbone. The search reward used in EfficientNetV2 combines the model accuracy, normalized training step time, and parameter size using a simple weighted product. The goal is to balance the trade-offs between accuracy, efficiency, and parameter size.

A mechanism named Progressive Learning is also introduced in EfficientNetV2 to gradually increase the resolution of the input images during training. The idea behind this technique is that it allows the model to learn the low-level features of the input images first, and then gradually increase the resolution to learn more complex features. This can make the training process more efficient and help the model converge faster. In practice, Progressive Learning is implemented by starting with low-resolution images and increasing the resolution over time. This can be done by progressively increasing the resolution coefficient, which controls the resolution of the input images. The progressive learning also help to reduce the overfitting that can happen when training a model on high resolution images. By starting with low-resolution images, the model can learn general features that are applicable to all resolutions, and then fine-tune these features as the resolution increases.

<img src="../assets/images/CNN/pandatrain_page-0001.jpg" width="600"/>

**Figure 1** Progressive Learning Progress. [Source](https://arxiv.org/abs/2104.00298)

There is an observation that we should also adjust the regularization strength accordingly to different image sizes. Progressive Learning with adaptive Regularization was proposed to address this insight byy gradually increasing the complexity of the network while also adjusting the regularization strength. There are 3 types of regularization techniques used in EfficientNet: Dropout, RandAugment, Mixup.

<img src="../assets/images/CNN/pandamixup_page-0001.jpg" width="600"/>

**Figure 2** Mixup Augmentation Technique. [Source](https://arxiv.org/abs/2104.00298)

<!-- The intuition of the scaling logic comes from 2 observations: 

- "Scaling up any dimension of network
width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models". This statement is saying that increasing any dimension of the network (width, depth, or resolution) can improve the accuracy of the model, but that this improvement becomes less significant as the model becomes larger. In other words, when you increase the width, depth or resolution of the network, you are adding more parameters and computations to the model which in turn can improve the accuracy of the model, but as the model becomes larger, the accuracy gain from adding more parameters and computations becomes less significant. This is due to the fact that larger models are more prone to overfitting, and the incremental benefit of adding more parameters and computations becomes smaller.

- "In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling." This statement is emphasizing the importance of balancing the dimensions of network width, depth, and resolution when scaling up a convolutional neural network (ConvNet) in order to achieve better accuracy and efficiency. -->

<!-- One of the key advantages of EfficientNet is that it can achieve state-of-the-art performance on a number of image classification benchmarks while also being more computationally efficient than previous models. This means that EfficientNet can be used to train large and complex models while still being able to run on devices with limited computational resources, such as smartphones or embedded devices.

Another important aspect of EfficientNet is that it can be easily scaled to different datasets and tasks. This is achieved through the use of a compound scaling method, which allows the model to be scaled up or down in a systematic and principled manner. This means that EfficientNet can be used for a wide range of image classification tasks, from small datasets to large-scale datasets, without the need for significant modifications to the model architecture. -->

Overall, EfficientNet is a powerful and versatile CNN architecture that has the potential to revolutionize the way we train and deploy large and complex models. Its ability to achieve state-of-the-art performance while also being computationally efficient makes it an ideal choice for a wide range of image classification tasks, from small datasets to large-scale datasets.

## 2.8 HRNet:

HRNet, short for High-Resolution Network, is a state-of-the-art deep learning model for image understanding tasks such as object detection, semantic segmentation, and human pose estimation. It was first introduced by a team of researchers at the Multimedia Laboratory of the Chinese University of Hong Kong led by Dr. Ke Sun. 

Before HRNet was published, the process of high-resolution recovery was typically achieved through the use of architectures such as Hourglass, SegNet, DeconvNet, U-Net, SimpleBaseline, and encoder-decoder networks. These architectures used a combination of upsampling and dilated convolutions to increase the resolution of the representations outputted by a classification or classification-like network. HRNet aims to improve upon these previous methods by introducing a new architecture that is specifically designed to learn high-resolution representations.

The observation that led to the idea of HRNet is that the existing state-of-the-art methods for these position-sensitive vision problems adopted the high-resolution recovery process to raise the representation resolution from the low-resolution representation outputted by a classification or classification-like network, which leads to loss of spatial precision. The researchers behind HRNet noticed that maintaining high-resolution representations throughout the entire process could potentially lead to more spatially precise representations and ultimately improve performance on these position-sensitive tasks.

The authors of the paper proposed a novel architecture that allows for the maintenance of high-resolution representations throughout the whole process. The network is composed of multiple stages, each stage contains multiple streams that correspond to different resolutions. The network performs repeated multi-resolution fusions by exchanging information across the parallel streams, allowing for the preservation of high-resolution information, and repeating multi-resolution fusions to boost the high-resolution representations with the help of the low-resolution representations.


<img src="../assets/images/CNN/traditionalone withhrnet.jpg" width="600"/>

**Figure 3** Traditional High Resolution Revovery (Above) Vs HRNet (Below). [Source](https://arxiv.org/abs/1908.07919)

HRNet maintain high-resolution representations throughout the network by starting with a high-resolution convolution stream as the first stage, and gradually adding high-to-low resolution streams one by one, forming new stages. The parallel streams at each stage consist of the resolutions from the previous stage, and an extra lower one, which allows for multi-resolution fusions and the ability to maintain high-resolution representations throughout the network. This architecture is called Parallel Multi-Resolution Convolutions.

Repeated Multi-Resolution Fusions is a technique used in the HRNet architecture to fuse representations from different resolution streams. This is done by repeatedly applying a transform function on each resolution stream, that is dependent on the input resolution index $x$ and the output resolution index r. The transform function is used to align the number of channels between the high-resolution and low-resolution representations. If the output resolution index ($r$) is lower than the input resolution index ($x$), the transform function (fxr(·)) downsamples the input representation (R) through $(r - x)$ stride-2 3 × 3 convolutions. For example, one stride-2 3 × 3 convolution for 2× downsampling, and two consecutive stride-2 3 × 3 convolutions for 4× downsampling. If the output resolution is higher than the input resolution, the transform function is upsampling the input representation R through the bilinear upsampling followed by a 1 × 1 convolution for aligning the number of channels.

<img src="../assets/images/CNN/representationhead.jpg" width="600"/>

**Figure 4** The representation head of HRNetV1, HRNetV2, HRNetV2p. [Source](https://arxiv.org/abs/1908.07919)

The resulting network is called HRNetV1, which is mainly applied to human pose estimation and achieves state-of-the-art results on COCO keypoint detection dataset. HRNetV2, on the other hand, combines the representations from all the high-to-low resolution parallel streams and is mainly applied to semantic segmentation, achieving state-of-the-art results on PASCAL-Context, Cityscapes, and LIP datasets. HRNetV2p is an extension of HRNetV2, which construct a multi-level representation and is applied to object detection and joint detection and instance segmentation. It improves the detection performance, particularly for small objects.

