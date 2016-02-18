---
title: Understanding Visual Concepts with Continuation Learning

author:
- name: William F. Whitney
- name: Michael Chang
- name: Tejas Kulkarni
- name: Joshua B. Tenenbaum
  affiliation: |
    | Department of Brain and Cognitive Science
    | Massachusetts Institute of Technology
    | `{wwhitney,mbchang,tejask,jbt}@mit.edu`

abstract: |
    In this paper, we introduce a neural network architecture and a learning algorithm to produce factorized symbolic representations. We propose to learn these concepts by observing consecutive frames, letting all the components of the hidden representation except a small discrete set (gating units) be predicted from previous frame, and let the factors of variation in the next frame be represented by discrete gating units (corresponding to symbolic representations). We demonstrate the efficacy of our approach on datasets of faces undergoing 3D transformations and Atari 2600 games.

bibliography: bibliography.bib
link-citations: yes
reference-section-title: References
---


# Introduction

<!--Deep learning has experienced great success in unsupervised learning representations of images that aid in the performance of tasks such as classification, object localization, and captioning. However, the representations produced by most deep learning techniques are highly entangled, without any notion of symbolic concepts. This makes these representations difficult to interpret or reuse as no single component of the representation vector has meaning by itself.

According to [@bengio2013representation], one key property of a good representation is that it should _disentangle the factors of variation_; that is, independent explanatory factors which occur together only occasionally should be represented distinctly. In this note we describe a model which is trained totally unsupervised, yet which is able to learn a factorization of videos into simple, easily interpretable concepts.
-->
Deep learning has led to remarkable breakthroughs in solving perceptual tasks such as object recognition, localization and segmentation using large amounts of labeled data. However, the problem of learning abstract representations of images without manual supervision is an open problem in machine perception. Existing unsupervised learning techniques have tried to address this problem [@hinton2006reducing; @ranzato2007unsupervised; @lee2009convolutional] but lack the ability to produce latent factors of variations or symbolic visual concepts from raw data. Computer vision has historically been formulated as the problem of producing symbolic descriptions of scenes from input images [@horn1986robot]. Without disentangled and symbolic visual concepts, it is difficult to interpret or re-use representations across tasks as no single component of the representation vector has a semantic meaning by itself. Traditionally, it has been difficult to adapt neural network architectures to learn such representations from raw data. In this paper, we introduce a neural network architecture and a learning algorithm to produce factorized symbolic representations given consecutive images. We demonstrate the efficacy of our approach on datasets of faces undergoing 3D transformations and Atari 2600 games.

## Related work

A number of generative models have been proposed in the literature to learn abstract visual representations including -- RBM-based models [@hinton2006reducing, @lee2009convolutional], variational based auto-encoders [@kingma2013auto,@rezende2014stochastic,@kulkarni2015deep], convolution based encoder-decoders [@ranzato2007unsupervised; @lee2009convolutional], and generative adversarial networks [@goodfellow2014generative; @radford2015unsupervised]. However, the representations produced by most of these techniques are entangled, without any notion of symbolic concepts. The exception to this is more recent work by Hinton et al. [@hinton2011transforming] on 'transforming auto-encoders' which use a domain-specific decoder with explicit visual entities to reconstruct input images. Inverse graphics networks [@kulkarni2015deep] have also been shown to disentangle interpretable factors of variations, albeit in a semi-supervised learning setting. Probabilistic program induction has been recently applied for learning visual concepts in the hand-written characters domain [@lake2015human]. However, this approach requires the specification of primitives to build up the final conceptual representations. The tasks we consider in this paper contain great conceptual diversity and it is unclear if there exist a simple set of primitives. Instead we propose to learn these concepts by observing consecutive frames, letting all the components of the hidden representation except a small discrete set (gating units) be predicted from previous frame, and let the factors of variation in the next frame be represented by discrete gating units (corresponding to symbolic representations).

<!-- [A representation that naturally extends / generalized itself to the temporal domain without explicit training on sequences] -->

# Model

![The gated model. Each frame encoder produces a representation from its input. The gating head examines both these representations, then picks one component from the encoding of time $t$ to pass through the gate. All other components of the hidden representation are from the encoding of time $t-1$. As a result, each frame encoder predicts what it can about the next frame and encodes the "unpredictable" parts of the frame into one component.](figures/model.pdf){ #fig:model width=100% }

This model is a deep convolutional autoencoder [@hinton2006reducing; @bengio2009learning; @masci2011stacked] with modifications to accommodate video prediction and encourage a particular factorization in the latent space. Given two frames in sequence, $x_{t-1}$ and $x_{t}$, the model first produces respective latent representations $h_{t-1}$ and $h_t$ through a shared encoder. The model then combines these two representations to produce a hidden representation $\tilde{h}_{t}$ that is fed as input to a decoder.

 We train the model using a novel objective function: given the previous frame $x_{t-1}$ of a video and the current frame $x_{t}$, reconstruct the current frame as $\hat{x}_{t}$.

$$\tilde{h}_{t} = Encoder(x_{t-1}, x_{t})$$

$$\hat{x}_{t} = Decoder(\tilde{h}_{t})$$

To produce $\tilde{h}_{t}$, we introduce a __gating__ in the encoder (see [@Fig:model]) that select a small set of __gating units__ that characterize the transformation <!-- better terminology --> between $x_{t-1}$ and $x_t$. For clarity, in this paper we describe our model under the context of one gating unit. Concretely, the encoder learns to use a _gating head_ that selects one index $i$ of the latent representation vector, and then $\tilde{h}_{t}$ is constructed as $h_{t-1}$, with the $i$th component of $h_{t-1}$ swapped out for the $i$th component of $h_t$.

Because the model must learn to reconstruct the current frame $t$ from a representation that is primarily composed of the components of the representation of $t-1$, the model is encouraged to represent the attributes of $t$ that are different from that of $t-1$, such as the lighting or pose of a face, in a very compact form which is completely disentangled from the invariant parts of the scene, such as the background. Thus, the model isolates the transformation from $t-1$ to $t$ from other latent features via the component $i$ selected by the gating head.



<!-- This forces the model to represent the events which are unpredictable, such as the action of an agent or a random event in a game, in a very compact form which is completely disentangled from the predictable parts of the scene, such as the background. -->


## Continuation Learning

To learn the gating function, we use a technique first described in [@whitney2016disentangled] for smoothly annealing a soft weighting function into a binary decision. Ordinarily, a model which produces a hard decision to gate through a single component (out of e.g. 200) would be difficult to train; in this case, it would require many forward passes through the decoder to calculate the expectation of the loss for each of the possible decisions. However, a model which uses a soft weighting over all the components can be trained with gradient descent in a single forward-backward pass.

In order to create a continuation between these two possibilities, we use a schedule of _weight sharpening_ [@graves2014neural] combined with noise on the output of the gating head. Given a weight distribution $w$ produced by the gating head and a sharpening parameter $\gamma$ which is proportional to the training epoch, we produce a sharpened and noised weighting:

$$w_i' = \frac{\big(w_i + \mathcal{N}(0, \sigma^2)\big)^{\gamma}}{\sum_j w_j^{\gamma}}$$

This formulation forces the gating head to gradually concentrate more and more mass on a single location at a time over the course of training, and in practice results in fully binary gating distributions by the end of training.



# Results

![**Manipulating the hidden representation.** Each row was generated by encoding an input image, then changing the value of a single component of the latent representation before rendering it with the decoder. **Top left:** a single unit controls the position of the paddle in Breakout. **Bottom left:** another unit controls the count of the remaining lives in the score bar. **Top right:** one unit controls the direction of lighting. **Middle right:** a unit that controls the elevation of the face. **Bottom right:** a unit controls the azimuth of the face, though this transformation is not smooth. All input images are from the test set.](figures/generalizations.png){ #fig:model width=100% }

## Atari frames

Our first dataset is frames from playing the Atari 2600 game Breakout. The model is given as input two frames which occurred in sequence, then reconstructs the second frame. This dataset was generated with a trained DQN network [@mnih2015human]. Since the model can only use a few components of its representation from the second frame, these components must contain all information necessary to predict the second frame given the first. For this dataset we use three gating heads, allowing three components of $h_t$ to be included in $\tilde{h}_t$.


## Synthetic faces

We trained the model on faces generated from the Basel face model and prepared as in [@kulkarni2015deep]. The input is two images of the same face between which only one of {lighting, elevation, azimuth} changes. For this dataset we use a single gating head, so the model must represent all differences between these two images in one unit only.


<!-- seeems like these are temporal "features" -->







<!-- References will be inserted automatically -->
