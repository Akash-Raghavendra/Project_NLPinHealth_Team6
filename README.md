# Automatic Phenotyping from Clinical Notes using MIMIC-IV

A project examining whether unstructured discharge summaries can be used to automatically identify patients with heart failure and diabetes, by fine-tuning a transformer-based clinical language model — built for a graduate NLP for Health course (CSCI 6907, GWU).

## Research Questions

1. Can unstructured discharge summaries accurately identify patients with heart failure and diabetes when compared to ICD-based phenotyping?
2. Which narrative sections within discharge summaries carry the strongest predictive signals for disease classification?

## Motivation

ICD diagnosis codes are widely used for research and quality measurement, but can suffer from incorrect entry, documentation inconsistencies, and variation in coding practice. Discharge summaries often reflect a fuller clinical picture — presenting symptoms, disease evolution, comorbidities, and treatment decisions — making them a promising source for identifying disease phenotypes directly from text.

## Approach

1. **Phenotype labeling** — generated binary labels for heart failure (ICD-10 code I50) and diabetes mellitus (codes E08–E13) from the MIMIC-IV diagnoses table
2. **Stratified sampling** — balanced the dataset across four subgroups (heart failure only, diabetes only, both, neither) to address class imbalance, yielding 128,776 records
3. **Section-based text preprocessing** — rather than using full discharge summaries, extracted only the clinically relevant sections (History of Present Illness, Past Medical History, Hospital Course, Discharge Diagnosis, Discharge Medications) to test which sections carry predictive signal
4. **Fine-tuning** — fine-tuned BioClinical ModernBERT (150M parameters, 4096-token context window, pre-trained on 53.5B+ clinical/biomedical tokens including prior MIMIC notes) on an NVIDIA A100 GPU
5. **Data validation** — ran structural inspection, PHI/raw-text detection, and tensor summary audits on all tokenized data given the sensitivity of clinical records
6. **Benchmarking** — compared results against a non-fine-tuned baseline, against ModernPubMedBERT, and against published F1 scores for other clinical embedding models

## Results

| Model | Macro F1 (before fine-tuning) | Macro F1 (after fine-tuning) |
|---|---|---|
| BioClinical ModernBERT | 0.18 | **0.69** |
| ModernPubMedBERT | 0.45 | 0.69 |

The fine-tuned BioClinical ModernBERT model outperformed the standard BioClinicalBERT benchmark (~0.59 F1 on similar tasks) and performed competitively with state-of-the-art retrieval-based embedding models (MedTE ~0.72, Nomic AI Embed Text ~0.69). The baseline (non-fine-tuned) model was essentially unable to detect heart failure at all, showing that fine-tuning — not just domain-pretrained knowledge — was the decisive factor in making phenotyping from narrative text work.

## Data & Ethics

- **Dataset:** MIMIC-IV (Medical Information Mart for Intensive Care IV), a publicly available, de-identified EHR dataset from MIT's Laboratory for Computational Physiology, used under its data usage agreement (CITI training completed by all team members)
- **Privacy:** MIMIC-IV is de-identified under HIPAA Safe Harbor; no re-identification was attempted, and automated PHI/raw-text scans were run on all processed data
- **Scope:** research and coursework only, not intended for clinical deployment

## Limitations

- Fine-tuned and evaluated on data from a single institution, limiting generalizability to other hospitals' documentation styles
- Not integrated into any live EHR workflow — real-world performance is untested
- Computational constraints limited the extent of hyperparameter tuning and cross-validation

## Tech Stack

Python · PyTorch · Hugging Face Transformers (BioClinical ModernBERT, ModernPubMedBERT) · Pandas

## Repository Structure

```
├── label_creation.py              # Binary phenotype labels from ICD-10 codes
├── sampling.py                    # Stratified sampling for class balance
├── text_preprocessing.py          # Section extraction and text cleaning
├── splitting_dataset.py           # Train/validation/test split
├── tokenization.py                # BioClinical ModernBERT tokenization
├── data_validation1.py            # Structural inspection
├── data_validation2.py            # PHI / raw-text detection
├── data_validation3.py            # Tensor summary audit
├── eda_diabetes_heartfailure.py   # Exploratory analysis of patient demographics
├── classification.py              # Model fine-tuning
├── pre_trained_eval.py            # Baseline (non-fine-tuned) model evaluation
├── fine_tuned_eval.py             # Fine-tuned model evaluation
├── Group5_CSCI6907_Final_Project_Report.pdf
├── Group5_CSCI6907_NLPinHealth_FinalProjectPresentation.pdf
└── README.md
```

*MIMIC-IV data is access-restricted and not included in this repo — available via [PhysioNet](https://physionet.org/content/mimiciv/) to credentialed researchers who complete the required data use agreement.*

## Team

Built by Akash Raghavendra, Nidhi Naidu, and Sai Srinivas for CSCI 6907: NLP for Health at The George Washington University.
