# Text segmentation and Pre-Processing

import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()  

print("Loading dataset from balanced_subset_stratified.csv")
result = pd.read_csv("Data/balanced_subset_stratified.csv")

print(f"Loaded {len(result):,} rows and {len(result.columns)} columns.")

# Text segmentation to keep only relevant sections
def extract_sections(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    pattern = r"(?im)(?P<header>[A-Z][A-Za-z\s]+)\s*:\s*(?P<content>.*?)(?=(?:[A-Z][A-Za-z\s]+ *:|$))"
    sections = {}

    for match in re.finditer(pattern, text, flags=re.DOTALL):
        header = match.group("header").strip().lower()
        content = match.group("content").strip()
        sections[header] = content

    keep = [
        "history of present illness",
        "past medical history",
        "hospital course",
        "discharge diagnosis",
        "discharge medications",
    ]

    filtered = {k: v for k, v in sections.items() if any(x in k for x in keep)}

    if not filtered:
        return text.strip()

    return "\n\n".join([f"{k}:\n{v}" for k, v in filtered.items()])



def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)    
    text = re.sub(r'\s+', ' ', text)          
    text = re.sub(r'_{2,}', ' ', text)       
    text = re.sub(r'\n+', '\n', text)       
    return text.strip()


def preprocess_dataset(input_df):
    df = input_df.copy()

    print("Extracting relevant sections")
    df["text"] = df["text"].progress_apply(extract_sections)

    print("Cleaning text")
    df["text"] = df["text"].progress_apply(clean_text)

    keep_cols = [
        "subject_id",
        "hadm_id",
        "diabetes_label",
        "heartfailure_label",
        "text",
    ]
    df = df[keep_cols].drop_duplicates(subset=["hadm_id"]).reset_index(drop=True)
    return df


print("Preprocessing")
processed = preprocess_dataset(result)
processed = processed.drop(['subject_id', 'hadm_id'], axis = 1)

processed.to_csv("Data/final_result_preprocessed.csv", index=False)

