# Reproducibility Environment

This document describes the software and hardware environment used for
the experiments reported in the DB-EpiFM paper.

## Core Software Environment

- Python: 3.11.7
- PyTorch: 2.1.2
- CUDA: 12.1

## Hardware

 NVIDIA GeForce RTX 4090 GPU with 24 GB memory.

## Installation

Create the Python environment:

```bash
conda create -n db-epifm python=3.11.7 -y
conda activate db-epifm
