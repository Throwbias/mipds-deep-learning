# MIPDS — Deep Learning Coursework

This repository contains my coursework, experiments, and
Multimodal Intelligent Perception and Decision System capstone.

## Project direction

Explore exercise-video analysis for coaching, beginning with
joint landmarks and estimated two-dimensional joint angles.

## Development environment

- Ubuntu 24.04 through WSL2
- Python 3.12
- TensorFlow 2.20.0
- NVIDIA GeForce RTX 4060 Ti

## Organization

- notebooks: hands-on activities and verification
- docs: design documents and written work
- experiments: experiment notes and results
- src: reusable Python code

## Current notebooks

The current vision prototype recognizes seven exercise types in frames from
public side-view recordings. Its recovered Milestone 4 training and Milestone 5
representation experiments are indexed in the [notebook guide](notebooks/README.md).
The [Architecture Record](docs/ARCHITECTURE_RECORD.md) documents the baseline,
including how it evolved from the planned pose workflow.

The next milestone evaluates the existing checkpoint on 570 held-out frames.
See the [Milestone 6 workflow](docs/VISION_RECOVERY_WORKFLOW.md) for the notebook,
required files, and report steps.
