# Dataset Documentation & Extraction Methodology

## 1. Original Dataset Overview

- **Dataset Name**: Indian Currency Dataset
- **Hosting Repository**: Mendeley Data
- **Version**: Version 1 (Published August 28, 2020)
- **DOI**: `10.17632/48ympv8jjf.1`
- **Original Source URL**: [https://data.mendeley.com/datasets/48ympv8jjf/1](https://data.mendeley.com/datasets/48ympv8jjf/1)
- **Contributors**: Venkataramana Veeramsetty, Gaurav Singal, Tapas Badal
- **Original Dataset Dimensions**: 11,657 images (4,657 raw camera captures + 7,000 augmented variations) across 7 denominations.
- **Original Archive Size**: **10.65 GB** (`10,658,785,013` bytes).
- **License / Terms of Use**: Please consult the original Mendeley Data page for current licensing and usage terms.

---

## 2. This Project's 350-Image Curated Subset

To build a lightweight, educational from-scratch CNN with fast local CPU training, we created a **curated subset of 350 images (50 images per class)**.

> [!IMPORTANT]
> This 350-image subset is an educational extraction from the complete 11,657-image Mendeley repository. It does not constitute the entire original archive.

### Key Dataset Differences
| Attribute | Original Mendeley Repository | This Project's Subset |
| :--- | :--- | :--- |
| **Total Images** | 11,657 images | **350 images** |
| **Images per Class** | ~1,000+ per denomination | **Exactly 50 per denomination** |
| **Raw Storage Size** | 10.65 GB | **~17.2 MB** (processed) |
| **Target Resolution** | $5344 \times 3006$ / $3006 \times 5344$ | **$64 \times 64$ RGB tensors** |

---

## 3. Extraction & Streaming Methodology

Rather than downloading the 10.65 GB full zip archive, [`prepare_dataset.py`](../prepare_dataset.py) streams the required samples directly via the Mendeley Data REST API:

1. **Class Category Discovery**: Queries `https://data.mendeley.com/public-api/datasets/48ympv8jjf/folders/1` with header `Accept: application/vnd.mendeley-public-dataset.1+json` to discover the 7 folder identifiers:
   - `Rs.10` (`79106392-976a-470e-aaf0-b84621794b54`)
   - `Rs.20` (`bd4d0b7b-b3c5-41a3-999d-54812a6a0081`)
   - `Rs.50` (`4e900bc4-d7f3-45d4-a159-c9c4bc55850d`)
   - `Rs.100` (`90b73257-aaa9-4c39-a803-189f262d04c4`)
   - `Rs.200` (`17b304df-a5bd-4c8a-9fdb-9bc72de3845e`)
   - `Rs.500` (`ccb77215-1c70-4228-aaf8-60c1717a83e5`)
   - `Rs.2000` (`6934acc5-a44a-48fa-8fa5-5337d68f36e8`)

2. **Deterministic File Selection**: For each class, file entries are queried and sorted **lexicographically by original filename** to eliminate random sampling and guarantee complete reproducibility.

3. **Integrity & Duplicate Verification**: Every downloaded byte stream is checked with Pillow (`img.verify()`, image decode, 3-channel RGB check) and deduplicated via SHA-256 hash tracking.

4. **Idempotent Local Caching**: Running [`prepare_dataset.py`](../prepare_dataset.py) again verifies existing files on disk without redownloading.

5. **Metadata Generation**: Generates [`dataset_metadata.json`](../dataset_metadata.json) mapping class names, integer labels (`0` to `6`), SHA-256 hashes, and dimensions.

---

## 4. Dataset Directory Hierarchy

```text
dataset/
├── Rs_10/
│   ├── image_001.jpg ... image_050.jpg
├── Rs_20/
│   ├── image_001.jpg ... image_050.jpg
├── Rs_50/
│   ├── image_001.jpg ... image_050.jpg
├── Rs_100/
│   ├── image_001.jpg ... image_050.jpg
├── Rs_200/
│   ├── image_001.jpg ... image_050.jpg
├── Rs_500/
│   ├── image_001.jpg ... image_050.jpg
└── Rs_2000/
    ├── image_001.jpg ... image_050.jpg
```

---

## 5. Statistical Validation & Partitioning

* **Total Dataset**: 350 images
* **Training Partition (80% Stratified)**: **280 images** (40 images/class)
* **Validation Partition (20% Stratified)**: **70 images** (10 images/class)
* **Zero Data Leakage**: $\text{Train} \cap \text{Val} = \emptyset$ (0 SHA-256 collisions).
* **Measured RGB Statistics (Train Set)**:
  - **Global Mean**: `0.4989` | **Global Std**: `0.1947`
  - **Red Channel**: $\mu = 0.5169, \sigma = 0.1886$
  - **Green Channel**: $\mu = 0.4943, \sigma = 0.1947$
  - **Blue Channel**: $\mu = 0.4855, \sigma = 0.1990$
