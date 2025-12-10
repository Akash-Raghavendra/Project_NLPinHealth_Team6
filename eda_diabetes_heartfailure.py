import pandas as pd

# Loading the data
def load_data(patients_path, admissions_path, diagnoses_path):
    patients = pd.read_csv("C:/Users/sais6/Desktop/Phenotyping_Project/Data/patients.csv")
    admissions = pd.read_csv("C:/Users/sais6/Desktop/Phenotyping_Project/Data/admissions.csv", parse_dates=['admittime'])
    diagnoses = pd.read_csv("C:/Users/sais6/Desktop/Phenotyping_Project/Data/diagnoses_icd.csv")
    return patients, admissions, diagnoses

# Filtering for our target ICD codes - {Heart Failure - I50, Diabetes Mellitus - E08, E09, E10, E11, E13}
def filter_by_icd(diagnoses, icd_prefixes=('I50', 'E8', 'E9', 'E08', 'E09', 'E10', 'E11', 'E13')):
    filtered = diagnoses[
        (diagnoses['icd_version'] == 10)
        & (diagnoses['icd_code'].str.startswith(icd_prefixes))
    ]
    return filtered

# Filtering only the unique subject id's
def get_target_subjects(filtered_diagnoses):
    return filtered_diagnoses['subject_id'].unique()

# Getting Patient level demographics
def filter_patients(patients, target_subjects):
    cols = ['subject_id', 'anchor_age', 'gender']
    return patients[patients['subject_id'].isin(target_subjects)][cols]

# Getting Hospital admission level demographics
def get_recent_admissions(admissions, target_subjects):
    admissions_filtered = admissions[admissions['subject_id'].isin(target_subjects)]
    recent = (
        admissions_filtered
        .sort_values('admittime')
        .groupby('subject_id')
        .last()
        .reset_index()
    )
    return recent[['subject_id', 'language', 'marital_status', 'race']]

# Merging all tables
def merge_patient_admission_info(patients_filtered, admissions_recent):
    df = pd.merge(patients_filtered, admissions_recent, on='subject_id', how='left')
    df.rename(columns={
        'anchor_age': 'age',
        'gender': 'gender',
        'language': 'language',
        'marital_status': 'marital_status',
        'race': 'race'
    }, inplace=True)

    df['language'].fillna('UNKNOWN', inplace=True)
    df['marital_status'].fillna('UNKNOWN', inplace=True)

    age_bins = [0, 18, 31, 41, 51, 61, 71, 81, 91, 151]
    age_labels = ['0-17', '18-30', '31-40', '41-50', '51-60',
                  '61-70', '71-80', '81-90', '91+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    return df

# Printing statistics
def generate_summary_statistics(df):
    print(df['age_group'].value_counts().sort_index())
    print(df['gender'].value_counts())
    print(df['language'].value_counts())
    print(df['marital_status'].value_counts())
    print(df['race'].value_counts())

# FInal dataset 
def build_final_dataset(patients_path, admissions_path, diagnoses_path):
    patients, admissions, diagnoses = load_data(patients_path, admissions_path, diagnoses_path)
    filtered_diag = filter_by_icd(diagnoses)
    target_subjects = get_target_subjects(filtered_diag)
    patients_filtered = filter_patients(patients, target_subjects)
    recent_admissions = get_recent_admissions(admissions, target_subjects)
    final_df = merge_patient_admission_info(patients_filtered, recent_admissions)
    return final_df


if __name__ == "__main__":
    final_df = build_final_dataset("patients.csv", "admissions.csv", "diagnoses_icd.csv")

    print(final_df.head())
    print(final_df.isnull().sum())
    generate_summary_statistics(final_df)

