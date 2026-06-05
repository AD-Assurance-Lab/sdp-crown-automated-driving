# AD-Assurance-Lab Dataset Setup

To run the automated driving neural network verification pipeline, you must independently acquire the raw datasets and place them exactly into the directory tree specified below.

## 1. Data Sources

### Udacity Self-Driving Car Dataset
* **Source:** [Kaggle - Udacity Self-Driving Car Behavioral Cloning](http://kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning)
* **Required Folders:** Extract the images (input) and csv file (output)

### ACDC (Automated Driving Dataset with Adverse Weather Conditions)
* **Source:** [ETH Zurich ACDC Download Page](https://acdc.vision.ee.ethz.ch/download)
* **Required Archives:** * `rgb_anon_trainvaltest.zip` (Main anonymous camera images)
  * `gt_trainval_ref.zip` (Ground truth references and semantic maps)

## 2. Expected Directory Tree
Extract and place the data matching this structure exactly:
```text
datasets/
├── README.md (This file)
├── ACDC/
│   ├── gt/
│   └── rgb_anon/
└── Udacity/
    ├── self_driving_car_dataset_jungle/
    └── self_driving_car_dataset_make/
