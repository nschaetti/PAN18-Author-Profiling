# PAN18-Author-Profiling

-----------------------------------------------------

Author Profiling Challenge of the PAN @ CLEF 2018

This repository consists of:

* find_alpha.py : Test various values for the leak parameter between image and tweet prediction.
* main.py : Use the model to predict the gender of Twitter profiles contained in a dataset.
* single_model_image.py : Train various image classification model for gender profiling.
* single_model_tweet.py : Train various tweet classification model for gender profiling.

Installation
============


PAN@CLEF18 Author profiling challenge
=====================================

Challenge introduction from https://pan.webis.de/clef18/pan18-web/author-profiling.html :

```
Authorship analysis deals with the classification of texts into classes based on the stylistic choices of
their authors. Beyond the author identification and author verification tasks where the style of individual
authors is examined, author profiling distinguishes between classes of authors studying their sociolect
aspect, that is, how language is shared by people. This helps in identifying profiling aspects such as
gender, age, native language, or personality type. Author profiling is a problem of growing importance
in applications in forensics, security, and marketing. E.g., from a forensic linguistics perspective
one would like being able to know the linguistic profile of the author of a harassing text message
(language used by a certain type of people) and identify certain characteristics (language as evidence).
Similarly, from a marketing viewpoint, companies may be interested in knowing, on the basis of the analysis
of blogs and online product reviews, the demographics of people that like or dislike their products. The
focus is on author profiling in social media since we are mainly interested in everyday language and how
it reflects basic social and personality processes.
```

## Task

This year the focus will be on gender identification in Twitter, where text and images may be used as information
sources. The languages addressed will be:

* English
* Spanish
* Arabic

## Training corpus

To develop your software, we provide you with a training data set that consists of Twitter users labeled with
gender. For each author, a total of 100 tweets and 10 images are provided. Authors are grouped by the language
of their tweets: English, Arabic and Spanish.

## Output

This software output for each document of the dataset a corresponding XML file that looks like this:

```
  <author id="author-id"
	  lang="en|es|ar"
	  gender_txt="male|female"
	  gender_img="male|female"
	  gender_comb="male|female"
  />
```

The software provide with three different predictions for the author's gender:

* gender_txt: gender prediction by using only text
* gender_img: gender prediction by using only images
* gender_comb: gender prediction by using both text and images

Image model training
====================

## Training


## Example


Tweet model training
====================

## The CNNCTweet model


## Training


## Example

Evaluation
==========

Other
=====

## Authors

## Citing

If you find my work useful for an academic publication, then please use the following BibTeX to cite it:

```
@misc{PAN18-Author-Profiling,
  author = {Schaetti, Nils},
  title = {UniNE at CLEF 2018: Character-based CNN and deep image classifier for gender profiling of Twitter users},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nschaetti/PAN18-Author-Profiling}},
}
```
