<!--# iStructTab (ICPR 2026 Submission)-->

# This Repository is under construction now

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" />
  <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C.svg" />
  <img src="https://img.shields.io/badge/modality-tabular+image-purple.svg" />
  <img src="https://img.shields.io/badge/status-single--blind%20review-orange.svg" />
  <img src="https://img.shields.io/badge/venue-ICPR%202026-black.svg" />
  <img src="https://img.shields.io/badge/Anonymous_Code-green.svg" />
</p>

> **Note (anonymized partial release).**  
> This repository contains an anonymized, **partial artifact** for the ICPR 2026 submission.  
> It includes the **GEDS-based iStructTab model**. 

---

## 1. Overview

**iStructTab** is a multimodal architecture for problems where each example has:

- **Tabular metadata** (numeric + categorical + optional text-like fields), and  
- **Image data** (e.g., medical images, natural images).

The key idea is to treat **both tabular features and image features as tokens**, then use **raph-Enhanced
Descriptor Sequencing (GEDS** to learn a global permutation over all tokens before feeding them to a transformer-like encoder (linformer).

This repository provides:

- `iStructTab` - a **simple GEDS-based variant** (default).
- A **training script** for one concrete dataset (HAM10000) using Optuna, mainly to:
  - Demonstrate how to plug iStructTab into a generic PyTorch pipeline.

iStructTab itself is **not specific** to HAM10000: any dataset with tabular + image inputs can be used by providing a matching PyTorch `Dataset` / `DataLoader`.

---

## 2. Repository Structure

A typical layout is:

```text
.
├── istructtab/
│   ├── __init__.py
│   ├── iStructTab.py              # GEDS + OEMT (default)
├── HAM_iStructTab.ipynb  # example on HAM10000
├── requirements.txt
└── README.md




