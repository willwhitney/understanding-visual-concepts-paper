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

This model is a deep convolutional autoencoder [@hinton2006reducing,@bengio2009learning,@masci2011stacked] with modifications to accomodate video prediction and encourage a particular factorization in the latent space.

$$\hat{x}_{t+1} = Decoder \big(Encoder(x_t, x_{t+1}) \big)$$


## Continuation learning

We use a technique first described in [@whitney2016disentangled] for smoothly annealing a soft weighting function into a hard decision.



## Related work

[@DBLP:journals/corr/LotterKC15]

# Results

## Atari frames

## Synthetic faces

## Bouncing balls

Dataset [@NIPS2008_3567]










<!-- References will be inserted automatically -->
