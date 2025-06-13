# PABT-Brain-Tumor-Clasification




## Overview

This project presents a comprehensive study on **brain tumor classification** using deep learning methods. The dataset comprises MRI images sourced from various public repositories including **Kaggle**, **Figshare**, and **BraTS**. Our approach leverages a hybrid architecture—**PABT (Proposed Attention-Based Transformer)**—that shows significant improvement in classification accuracy across multiple tumor types.

---

## Pie Chart Interpretation

The pie chart above visualizes **gender-wise distribution of brain tumor cases and related mortality**:

- **New Cases (Men)** – 33%
- **New Cases (Women)** – 25%
- **Deaths (Men)** – 24%
- **Deaths (Women)** – 18%

This indicates a slightly higher incidence and mortality in men compared to women, underscoring the importance of tailored screening and treatment strategies.

---

## Research Evaluation: Performance Comparison (𝛼%)

| **RESEARCH**                             | **DATA SOURCE** | **𝛼 (%)** |
|------------------------------------------|------------------|-----------|
| Sharif et al., 2022                      | Figshare         | 96.80     |
| Swati et al., 2019                       | Figshare         | 94.82     |
| Sajjad et al., 2019                      | Figshare         | 90.67     |
| Tabatabaei et al., 2023                  | BraTS            | 96.30     |
| Deepak & Ameer, 2019                     | BraTS            | 95.60     |
| Özkaraca et al., 2023                    | Kaggle           | 96.00     |
| Rahman & Islam, 2023                     | Kaggle           | 98.12     |
| Muezzinoglu et al., 2023                 | Kaggle           | 98.10     |
| Ali et al., 2023                         | Kaggle           | 95.70     |
| Aloraini et al., 2023                    | Kaggle           | 96.95     |
| ZahraaAlaatageldein et al., 2024         | Kaggle           | 94.00     |
| **PABT (Proposed)**                      | Kaggle           | **98.40** |

> **𝛼**: Classification accuracy (%)  
> **Note**: PABT outperforms existing models in terms of classification performance on the Kaggle dataset.

---

## Model Highlights

- **Architecture**: CNN + Transformer + Attention-based architecture.
- **Backbones Used**: EfficientNet, ResNet, DenseNet, VGG, Vision Transformers.
- **Residual Connections**: Integrated into each backbone for stable gradient flow.
- **Input Data**: Preprocessed T1-weighted brain MRIs.
- **Output Classes**: Glioma, Meningioma, Pituitary Tumor.

---

## Dataset Sources

- **Kaggle**: Brain Tumor MRI datasets ([Link](https://www.kaggle.com))
- **Figshare**: Multiple open datasets for glioma/meningioma ([Link](https://figshare.com))
- **BraTS**: Benchmark dataset for brain tumor segmentation and classification ([Link](https://www.med.upenn.edu/cbica/brats2020/data.html))

---

## Evaluation Metrics

- **Accuracy**  
- **Precision, Recall, F1-score** (Per class)
- **Cohen’s Kappa & Jaccard Index**
- **Confusion Matrix Visualization**
- **ROC-AUC**

---

## Tools & Frameworks

- **TensorFlow / Keras**: Model development and training.
- **Scikit-learn**: Evaluation and statistical analysis.
- **Matplotlib & Seaborn**: Plotting confusion matrices, loss/accuracy curves.
- **OpenCV / PIL**: Preprocessing MRI images.

---

## Conclusion

Our proposed **PABT** model sets a new benchmark for automated brain tumor classification using MRI data. By integrating multiple deep learning backbones and attention mechanisms, we achieve improved performance compared to state-of-the-art literature.

---

## Citation

If you use this project in your research, please cite:

> T. Banerjee, *PABT: Proposed Attention-Based Transformer for Brain Tumor Classification*, 2025.

---

## License

This project is open-source under the MIT License.

---

## Contact

For questions or collaborations, feel free to reach out to [Tathagat Banerjee](mailto:tathagatbanerjee@example.com).
