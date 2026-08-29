# OmniCare

Source code for our paper "OmniCare: A multimodal framework bridging molecular, tissue, and clinical data for generalizable cancer outcome prediction".

## Preparing data

### Data source
TCGA data are available at [images](https://portal.gdc.cancer.gov), [demographics and genomics](https://www.cbioportal.org), and [reports](https://github.com/cpystan/Wsi-Caption), respectively. [CPTAC](https://portal.gdc.cancer.gov/projects/CPTAC-3), [HistAI](https://www.hist.ai), [HANCOCK](https://hancock.research.fau.eu), [SurGen](https://github.com/CraigMyles/SurGen-Dataset).

### Software

[DeepSeek-Chat](https://www.deepseek.com), [Qwen3-embedding-0.6B](https://github.com/QwenLM/Qwen3-Embedding), [CONCH](https://github.com/mahmoodlab/CONCH), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [scFoundation](https://github.com/biomap-research/scFoundation), [Vision Transformer](https://github.com/google-research/vision_transformer)


## Running 
```python
# Pre-training
python pretrain.py OmniCare 

# Fine-tuning
python finetune.py OmniCare --cohort=[cohort name]

# Validation
python test.py OmniCare

```
