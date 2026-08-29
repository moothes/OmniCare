# OmniCare

Source code for our paper "OmniCare: A multimodal framework bridging molecular, tissue, and clinical data for generalizable cancer outcome prediction".

## Preparing data

### Data source
--TCGA data, including pathology images, demographics, mRNA expression profiles, and pathology reports, are available at [images](https://portal.gdc.cancer.gov), [demographics and genomics](https://www.cbioportal.org), and [reports](https://github.com/cpystan/Wsi-Caption), respectively. 

--CPTAC data can be accessed at [here](https://portal.gdc.cancer.gov/projects/CPTAC-3). 

--The HistAI dataset is available at [here](https://www.hist.ai). 

--HANCOCK is accessible at [here](https://hancock.research.fau.eu). 

--SurGen is available at [here](https://github.com/CraigMyles/SurGen-Dataset).

### Software

--DeepSeek-Chat from [here](https://www.deepseek.com) 

--Qwen3-embedding-0.6B from [here](https://github.com/QwenLM/Qwen3-Embedding)

--CONCH from [here](https://github.com/mahmoodlab/CONCH)

--Virchow2 from [here](https://huggingface.co/paige-ai/Virchow2)

--scFoundation from [here](https://github.com/biomap-research/scFoundation)

--Vision Transformer from [here](https://github.com/google-research/vision_transformer)


## Running 
```python
# Pre-training
python pretrain.py OmniCare 

# Fine-tuning
python finetune.py OmniCare --cohort=[cohort name]

# Validation
python test.py OmniCare

```
