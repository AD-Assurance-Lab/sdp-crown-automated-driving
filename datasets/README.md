# AD-Assurance-Lab Dataset Setup

To run the automated driving neural network verification pipeline, you must acquire the raw datasets independently and structure them exactly as shown below.

## 1. Data Sources
* **ACDC Dataset:** Download the generic tracking and ground truth segments from the official ACDC website.
* **Udacity Self-Driving Car Dataset:** Pull the steering camera captures and dataset packages directly from the Udacity repository/Kaggle mirrors.

## 2. Expected Directory Tree
Place the extracted data inside this `datasets/` folder matching this layout:
```text
datasets/
├── ACDC/
│   ├── gt/
│   └── rgb_anon/
└── Udacity/
    ├── self_driving_car_dataset_jungle/
    └── self_driving_car_dataset_make/
