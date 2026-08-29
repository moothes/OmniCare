# OmniCare

Source code for our paper "OmniCare: A multimodal framework bridging molecular, tissue, and clinical data for generalizable cancer outcome prediction".

## Preparing data and software

### Data source
TCGA data are available at [images](https://portal.gdc.cancer.gov), [demographics and genomics](https://www.cbioportal.org), and [reports](https://github.com/cpystan/Wsi-Caption), respectively. [CPTAC](https://portal.gdc.cancer.gov/projects/CPTAC-3), [HistAI](https://www.hist.ai), [HANCOCK](https://hancock.research.fau.eu), [SurGen](https://github.com/CraigMyles/SurGen-Dataset).

### Software
[DeepSeek-Chat](https://www.deepseek.com), [Qwen3-embedding-0.6B](https://github.com/QwenLM/Qwen3-Embedding), [CONCH](https://github.com/mahmoodlab/CONCH), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [scFoundation](https://github.com/biomap-research/scFoundation), [Vision Transformer](https://github.com/google-research/vision_transformer)

## Reproducing process for reviewers
To ensure the reproducibility of our code for reviewer to validate OmniCare, we provide an introduction on verifying our model during reviewing in journals. 

1. Downloading the extracted features from downstream cohorts, including pathology images, pathology reports, demographics, and genomics data at [Coming soon](https://github.com).

Downloading the pretrained weights from [Coming soon](https://github.com).
  
Then, putting the downloaded file at the root path of this project like ```./downstream/[cohort_task]```.

2. Runing the following code:
```python
python finetune.py OmniCare --cohort=[cohort_task]
```
```cohort_task``` should one of the folder name in ```./downstream```. 

```Cohort``` and ```task``` indicate the cohort name and specific endpoint, such as overall survival (OS), disease-free survival (DFS, recurrence), metastasis detection (MD).

3. The results will be saved under ```./results/OmniCare/finetune/```

## Running 
```python
# Pre-training
python pretrain.py OmniCare 

# Fine-tuning
python finetune.py OmniCare --cohort=[cohort name]

# Validation
python test.py OmniCare

```
