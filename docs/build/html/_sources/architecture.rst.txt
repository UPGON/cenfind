Architecture
============

The general architecture of Cenfind consists of the following modules:

Data module
-----------
deals with the underlying data from the file system

Detection module
----------------
defines the different detectors with their potentially associated model weights.

Measurement module
------------------
measures the accuracy of the different predictions

Structure module
----------------
defines the types of objects created by the detectors such as Points for centrioles and cilia and Contours for nuclei

Visualisation module
--------------------
helps produce visual representation of the detection and assignment procedures for quality control

Serialisation module
--------------------
describes how the outputs need to be saved to the file system
