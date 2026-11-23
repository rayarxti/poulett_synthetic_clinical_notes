import asyncio
import json

import pandas as pd
import pytest
import pytest_asyncio

from src.data_generator import DSM5_CRITERION_ORDER, generate_admissions
from src.dataset_utils import prepare_admission_data

admission_generator = generate_admissions(elective_admission_rate = 0)

def test_chief_complaint_generation():
    """
    Confirm that age, chief complaint and diagnosis are generated correctly
    as part of admission complaint generation
    """
    admission_details = admission_generator.generate_emergency_admission_complaint("M", 77)
    
    assert len(admission_details["chief_complaint"]) > 0
    assert len(admission_details["diagnosis"]) > 0


def test_admission_matches_trajectory_visit_and_complaint(monkeypatch):
    status = [False] * len(DSM5_CRITERION_ORDER)
    status[3] = True
    patient = {
        "patient_id": "nsduh-test",
        "name": "Jane Smith",
        "age": 32,
        "gender": "Female",
        "medical_record_number": "123456789",
        "nhs_number": "987654321",
        "target_substance": "alcohol",
        "visit_months": [3],
        "trajectory": [status],
        "dsm5_criteria": {criterion: "No" for criterion in DSM5_CRITERION_ORDER},
    }
    generator = generate_admissions(
        patients=pd.DataFrame([json.dumps(patient)]),
        elective_admission_rate=1.0,
        number_of_generations=1,
    )

    async def fake_call_llm_async(prompt, model):
        if "single integer output" in prompt:
            return "3"
        assert "alcohol use with DSM-5 criteria endorsed at month 3" in prompt
        assert "C4 - Craving or strong urge to use" in prompt
        assert "2026-03-01" in prompt
        return json.dumps({
            "date": "2026-03-01",
            "time": "09:00",
            "method": "A&E",
            "admission_type": "emergency",
            "chief_complaint": "Chest pain",
            "ED_diagnosis": "Chest pain",
        })

    monkeypatch.setattr("src.data_generator.call_llm_async", fake_call_llm_async)
    admissions = asyncio.run(generator.run(return_output=True))
    admission = json.loads(admissions.iloc[0, 0])

    assert admission["trajectory_index_date"] == "2025-12-01"
    assert admission["trajectory_visit_month"] == 3
    assert admission["trajectory_visit_date"] == admission["date"] == "2026-03-01"
    assert admission["trajectory_status"] == status
    assert admission["positive_dsm5_criteria"] == [
        "C4 - Craving or strong urge to use"
    ]
    assert "C4 - Craving or strong urge to use" in admission["chief_complaint"]

    patient_record = {
        **patient,
        "date_of_birth": "1994-01-01",
        "medical_record_number": "123456789",
        "admission_details": {
            **admission,
            "admission_id": "admission-test",
            "encounter_id": "encounter-test",
            "ward": "Medical Ward",
            "bed_location": "B01",
        },
    }
    admission_data = prepare_admission_data(patient_record, version="test")
    assert admission_data["trajectory_visit_month"] == 3
    assert admission_data["trajectory_status"] == status
    assert admission_data["positive_dsm5_criteria"] == [
        "C4 - Craving or strong urge to use"
    ]