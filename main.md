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
    We should put an abstract here...

bibliography: bibliography.bib
link-citations: yes
reference-section-title: References
---


# Introduction

Deep learning has experienced great success in unsupervised learning representations of images that aid in the performance of tasks such as classification, object localization, and captioning. However, the representations produced by most deep learning techniques are highly entangled, without any notion of symbolic concepts. This makes these representations difficult to interpret or reuse as no single component of the representation vector has meaning by itself.

According to [@bengio2013representation], one key property of a good representation is that it should _disentangle the factors of variation_; that is, independent explanatory factors which occur together only occasionally should be represented distinctly. In this note we describe a model which is trained totally unsupervised, yet which is able to learn a factorization of videos into simple, easily interpretable concepts.

# Model

![The gated encoder. Each frame encoder produces a representation from its input. The gating head examines both these representations, then picks one component from the encoding of time $t$ to pass through the gate. All other components of the hidden representation are from the encoding of time $t-1$. As a result, each frame encoder predicts what it can about the next frame and encodes the "unpredictable" parts of the frame into one component.](figures/encoder.pdf){ #fig:encoder width=60% }

This model is a deep convolutional autoencoder [@hinton2006reducing; @bengio2009learning; @masci2011stacked] with modifications to accomodate video prediction and encourage a particular factorization in the latent space. We train the model using a novel objective function: given a the previous frame of a video and the current frame, reconstruct the current frame. 

$$\hat{x}_{t} = Decoder \big(Encoder(x_{t-1}, x_{t}) \big)$$

We also introduce a _gating_ in the encoder (see [@Fig:encoder]) such that all components of the hidden representation except one must be predicted from the previous frame, and only one component of the current frame's encoding is used. This forces the model to represent the events which are unpredictable, such as the action of an agent or a random event in a game, in a very compact form which is completely disentangled from the predictable parts of the scene, such as the background.


## Continuation learning

We use a technique first described in [@whitney2016disentangled] for smoothly annealing a soft weighting function into a binary decision. Ordinarily, a model which produces a hard decision to gate through a single component (out of e.g. 200) would be difficult to train; in this case, it would require many forward passes through the decoder to calculate the expectation of the loss for each of the possible decisions. However, a model which uses a soft weighting over all the components can be trained with gradient descent in a single forward-backward pass.

In order to create a continuation between these two possibilities, we use a scheduling for _weight sharpening_ [@graves2014neural] combined with noise on the output of the gating head. Given a weight distribution $w$ produced by the gating head and a sharpening parameter $\gamma$ which is proportional to the training epoch, we produce a sharpened and noised weighting:

$$w_i' = \frac{\big(w_i + \mathcal{N}(0, \sigma^2)\big)^{\gamma}}{\sum_j w_j^{\gamma}}$$

This formulation forces the gating head to gradually concentrate more and more mass on a single location at a time over the course of training, and in practice results in fully binary gating distributions by the end of training.

## Related work

[@DBLP:journals/corr/LotterKC15]

# Results

## Atari frames

## Synthetic faces

## Bouncing balls

Dataset [@NIPS2008_3567]










<!-- References will be inserted automatically -->
