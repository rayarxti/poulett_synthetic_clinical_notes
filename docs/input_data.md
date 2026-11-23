# Input Data

There are three input tables that need to be passed to the pipeline:

1. `elective_admissions_dataset`
2. `emergency_admissions_dataset`
3. `patients_input_dataset`

Examples of each table are included in the repo. These were used when testing the pipeline so should create realistic notes.

Details regarding each dataset are documented below:

## Elective Admissions Dataset

This dataset contains the columns:

- **Age_Category**: String of form 'x-y' where x and y are number representing the lower bound and higher bound of an age respectively.
- **Sex**: String of 'Male' or 'Female'.
- **Speciality**: String containing speciality for the admission. e.g. 'Orthopaedics'.
- **Procedure**: String containing a short description of the procedure.

❇️ **Tip** ❇️ If appropriate, include 'left' or 'right' in the procedure description.

## Emergency Admissions Dataset

- **Age_Category**: String of form 'x-y' where x and y are number representing the lower bound and higher bound of an age respectively.
- **Sex_Category**: String of 'Male' or 'Female'.
- **ChiefComplaintDescription**: String containing description of chief complaint.
- **DiagnosisDescription**: String containing description of the diagnosis.
- **rare_disease**: Integer `1` or `0` flagging if disease is rare.
- **NovelDiseaseFlag**: Integer `1` or `0` flagging if disease is novel.
- **AdditionalSymptoms**: String or `None` containing details on additional symptoms - useful for novel diseases.
- **AdditionalInformation**: String or `None` containing additional information. Useful for novel diseases.
- **ConfirmedBy**: String or `None` containing details on what confirmed the diagnosis, e.g. a swab. Useful for novel diseases.
- **SupportedBy**: String ot `None` containing details on information supporting the diagnosis, e.g. a rash. Useful for novel diseases.

❇️ **Tips** ❇️

- If appropriate, include 'left' or 'right' in descriptions.

## Patients Input Dataset

- **FIRST**: String of first name.
- **MIDDLE**: String of middle name.
- **LAST**: String of last name.
- **RACE**: String of race. Not currently used in pipeline but could be used for evaluation purposes.
- **ETHNICITY**:  String of ethnicity. Not currently used in pipeline but could be used for evaluation purposes.
- **GENDER**: String of gender. `M` or `F`.
- **ADDRESS**: String of synthetic address.
- **CITY**: String of city.
- **POSTCODE**: String of first half of postcode. e.g. `PL1`.

## NSDUH Synthetic Training Profiles

The patient generator samples records from the NSDUH training split and
preserves each record's demographics, DSM-5 criteria, and substance-use
trajectory in the patient intermediates. Configure
`nsduh_training_profiles_path` in `config/params.py` to point to
`synthetic_output/train.json`; relative paths resolve from the Poulett project
root.

Patient records are built directly from the sampled profiles, without an LLM
patient-generation call. Fields missing from the profiles, including names,
addresses, contact details, GP, and next of kin, are filled using the existing
patients input dataset and synthetic values.

Profiles are sampled without replacement within the configured
`minimum_patient_age` through `maximum_patient_age` range. The default lower
bound excludes younger profiles because the emergency-admission inputs used by
this pipeline do not cover patients below 18. Names are sampled from the
patients input dataset with sex matched to the profile.

For each sampled patient, one visit month is selected and its aligned 11-item
trajectory status determines the admission context. Visit dates are calculated
as calendar months after `nsduh_index_date` in `config/params.py`. Any positive
DSM-5 criteria are appended to the chief complaint and recorded with the
admission; positive-criteria visits are generated as emergency admissions.

## Setting Up

This pipeline was tested in Foundry. Therefore, tables were read and written using the `read_write_data` function.

Please see the `changing_functions.md` document for more details.
