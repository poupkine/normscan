import pydicom

def extract_study_series_uids(dicom_datasets: List[pydicom.Dataset]) -> dict:
    """
    Извлекает StudyInstanceUID и SeriesInstanceUID из первого среза.
    """
    ds = dicom_datasets[0]
    return {
        'study_uid': ds.StudyInstanceUID,
        'series_uid': ds.SeriesInstanceUID
    }