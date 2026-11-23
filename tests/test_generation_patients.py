import asyncio
import json

import pandas as pd

from src.data_generator import generate_patients
from src.dataset_utils import prepare_patient_data
from src.schemas import patient_schema

patient_generator = generate_patients()

def test_generate_allergies():
    """
    Test the generation of allergies.
    """
    
    allergy_prevalence = {
    "pollen" : 0, 
    }
    
    assert patient_generator.generate_patient_allergies(allergy_prevalence) == "None"
    
    allergy_prevalence = {
    "pollen" : 1, 
    }
    
    assert patient_generator.generate_patient_allergies(allergy_prevalence) == "pollen"
    
    allergy_prevalence = {
    "pollen" : 1,
    "nuts" : 1
    }
    
    allergies = patient_generator.generate_patient_allergies(allergy_prevalence)
    
    assert "nuts" in allergies
    assert "pollen" in allergies


def test_generate_patients_uses_nsduh_demographics_and_status(monkeypatch):
    profile = {
        "synthetic_id": "SYN-TEST",
        "target_substance": "alcohol",
        "other_substances_used": ["cannabis"],
        "substance_subclass": "Whiskey",
        "visit_months": [1, 4],
        "trajectory": [[True, False], [False, False]],
        "profile": {
            "demographics": {
                "Age": 32,
                "Sex": "Female",
                "Race": "Hispanic",
                "Education": "Some college credit, but no degree",
            },
            "dsm5_criteria": {"C1 - Larger amounts or longer than intended": "Yes"},
        },
    }

    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("Patient profiles should not be generated with an LLM")

    monkeypatch.setattr("src.data_generator.call_llm_async", unexpected_llm_call)
    names_df = pd.DataFrame([
        {
            "FIRST": "Jane",
            "MIDDLE": "A",
            "LAST": "Smith",
            "GENDER": "F",
            "ADDRESS": "1 Example Road",
            "CITY": "Exampletown",
            "POSTCODE": "EX1",
        },
        {
            "FIRST": "John",
            "MIDDLE": "B",
            "LAST": "Doe",
            "GENDER": "M",
            "ADDRESS": "2 Example Road",
            "CITY": "Exampletown",
            "POSTCODE": "EX1",
        },
    ])
    generator = generate_patients(
        names_df=names_df,
        number_of_generations=1,
        profiles=[profile],
    )

    patients_df = asyncio.run(generator.run(return_output=True))
    patient = json.loads(patients_df.iloc[0, 0])

    assert patient["age"] == 32
    assert patient["gender"] == "Female"
    assert patient["name"] == "Jane A Smith"
    assert patient["nsduh_synthetic_id"] == "SYN-TEST"
    assert patient["demographics"] == profile["profile"]["demographics"]
    assert patient["dsm5_criteria"] == profile["profile"]["dsm5_criteria"]
    assert patient["target_substance"] == "alcohol"
    assert patient["trajectory"] == profile["trajectory"]
    assert patient["date_of_birth"] == "1994-01-01"
    assert patient["contact_number"].startswith("07700 900")
    assert patient["next_of_kin"]["name"]
    assert patient["next_of_kin"]["relationship"]
    assert patient["gp_details"]["name"].endswith("(GP)")
    assert patient["gp_details"]["practice_name"]
    assert patient["medical_record_number"]
    assert patient["nhs_number"]

    patient_data = prepare_patient_data(patient)
    assert patient_data["demographics"] == profile["profile"]["demographics"]
    assert patient_data["dsm5_criteria"] == profile["profile"]["dsm5_criteria"]
    patient_table = pd.DataFrame([patient_data]).astype(patient_schema)
    assert patient_table.at[0, "dsm5_criteria"] == profile["profile"]["dsm5_criteria"]