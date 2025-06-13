# PABT-Brain-Tumor-Clasification
PABT Brain Tumor Clasification


brain-tumor-pabt/
│
├── 📁 data/

│   ├── README.md             # Instructions for dataset structure

│   └── (Placeholders for Figshare, Br35H, Sartaj datasets)

│
├── 📁 models/

│   ├── nasnet_ann.py

│   ├── nasnet_dnn.py

│   ├── nasnet_cnn.py

│   ├── nasnet_lstm.py

│   ├── nasnet_cnn_lstm.py

│   ├── nasnet_ca_cnn_dnn.py

│   └── pabt.py               # Final NASNet + PABT Model

│
├── 📁 utils/

│   ├── preprocessing.py

│   ├── metrics.py

│   ├── train_utils.py

│   └── visualization.py      # Confusion matrix, ROC curves, heatmaps



│
├── 📁 results/

│   └── fold_scores.csv       # Scores from 9-fold CV

│
├── main.py                   # Main training script

├── requirements.txt          # All libraries and versions
├── README.md                 # Overview, model architecture, usage
└── LICENSE
