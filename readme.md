# OmniCare

Source code for our paper "OmniCare: A multimodal framework bridging molecular, tissue, and clinical data for generalizable cancer outcome prediction".

## Reproducing process for reviewers
To ensure the reproducibility of our work and facilitate the validation process for reviewers, we provide a dedicated guide in this README for verifying the OmniCare model. To minimize the overhead associated with data preprocessing, we also release the pre-extracted features of our private cohorts, allowing reviewers to directly focus on model evaluation.

Step 1: Download prerequisite data
Extracted features for downstream cohorts are available at [Coming soon](https://github.com).
Pretrained OmniCare weights are available at [pretrained weights](https://drive.google.com/file/d/1dB0EKSvyn3WNSAVgfksO8oFQydK2NPIZ/view?usp=sharing).
Place all downloaded files into the corresponding cohort task folder under the project root like ```./downstream/[cohort_task]```.

3. Runing the following code:
```python
python finetune.py OmniCare --cohort=[cohort_task]
```
```cohort_task``` should one of the folder name in ```./downstream```. 

```Cohort``` and ```task``` indicate the cohort name and specific endpoint, such as overall survival (OS), disease-free survival (DFS, recurrence), metastasis detection (MD).

3. The results will be saved under ```./results/OmniCare/finetune/```

## Preparing data and software

### Software
[DeepSeek-Chat](https://www.deepseek.com), [Qwen3-embedding-0.6B](https://github.com/QwenLM/Qwen3-Embedding), [CONCH](https://github.com/mahmoodlab/CONCH), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [scFoundation](https://github.com/biomap-research/scFoundation), [Vision Transformer](https://github.com/google-research/vision_transformer)

### Data source
TCGA data are available at [images](https://portal.gdc.cancer.gov), [demographics and genomics](https://www.cbioportal.org), and [reports](https://github.com/cpystan/Wsi-Caption), respectively. [CPTAC](https://portal.gdc.cancer.gov/projects/CPTAC-3), [HistAI](https://www.hist.ai), [HANCOCK](https://hancock.research.fau.eu), [SurGen](https://github.com/CraigMyles/SurGen-Dataset).

### Data preprocessing
While most modalities can be fed into OmniCare directly, the main modality needs further processing is pathology data. We leverage the patching and feature extracting tools in [PrePath](https://github.com/birkhoffkiki/PrePATH) to convert raw whole slide images into feature embeddings.

## Running 
```python
# Pre-training
python pretrain.py OmniCare 

# Fine-tuning
python finetune.py OmniCare --cohort=[cohort name]

# Validation
python test.py OmniCare

```
