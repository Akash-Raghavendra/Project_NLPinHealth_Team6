import pandas as pd
from eda_diabetes_heartfailure import load_data

def load_summaries(summaries_path):
    summaries = pd.read_csv(summaries_path)
    return summaries

def combining_tables():
    patients_path = "Data/patients.csv"
    admissions_path = "Data/admissions.csv"
    diagnoses_path = "Data/diagnoses_icd.csv"
    summaries_path = "Data/discharge.csv"

    patients, admissions, diagnoses = load_data(patients_path, admissions_path, diagnoses_path)
    summaries = load_summaries(summaries_path)

    diag_grouped = (
        diagnoses.groupby(['subject_id', 'hadm_id'])
        .agg({
            'icd_code': list,
            'icd_version': list
        })
        .reset_index()
    )

    adm_diag = pd.merge(admissions, diag_grouped, on=['subject_id', 'hadm_id'], how='left')
    all_combined = pd.merge(adm_diag, summaries, on=['subject_id', 'hadm_id'], how='left')

    return all_combined


def creating_labels():
    all_combined = combining_tables()

    all_combined['icd_version'] = all_combined['icd_version'].apply(
        lambda x: [int(v) for v in x if str(v).isdigit()] if isinstance(x, list) else []
    )

    def label_condition(codes, versions):
        if not isinstance(codes, list) or not isinstance(versions, list):
            return pd.Series({'heartfailure_label': 0, 'diabetes_label': 0})
        icd10_codes = [c for c, v in zip(codes, versions) if v == 10]
        heartfailure = any(c.startswith('I50') for c in icd10_codes)
        diabetes = any(c.startswith(('E8', 'E9', 'E08', 'E09', 'E10', 'E11', 'E13')) for c in icd10_codes)
        return pd.Series({'heartfailure_label': int(heartfailure), 'diabetes_label': int(diabetes)})

    labels = all_combined.apply(
        lambda row: label_condition(row['icd_code'], row['icd_version']), axis=1
    )

    all_combined = pd.concat([all_combined, labels], axis=1)

    return all_combined


def final_table():
    all_combined = creating_labels()

    keep_cols = [
        'subject_id',
        'hadm_id',
        'admittime',
        'dischtime',
        'text',
        'diabetes_label',
        'heartfailure_label',
        'icd_code',
        'icd_version'
    ]

    all_combined_cleaned = all_combined[keep_cols].reset_index(drop=True)
    return all_combined_cleaned

result = final_table()
result["text"] = result["text"].fillna("").astype(str)
print(result.head())
print(result.columns)
result.to_csv('Data/output_table.csv', index=False)
