from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import json
from ml.bert_inference import get_bert_prediction
from parsers.hh_document_parser import parse_hh_pdf
from parsers.hh_link_parser import parse_hh_link

app = FastAPI()

session_data = {}


@app.post('/upload-pdf/')
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(...)):
    contents = await file.read()
    with open('temp_resume.pdf', 'wb') as f:
        f.write(contents)
    data = parse_hh_pdf('temp_resume.pdf')
    session_data[session_id] = data
    return JSONResponse(content={'status': 'success', 'data': data})


@app.post('/process-hh-link/')
async def process_hh_link(link: str = Form(...), session_id: str = Form(...)):
    data = parse_hh_link(link)
    session_data[session_id] = data
    return JSONResponse(content={'status': 'success', 'data': data})


@app.post('/upload-json/')
async def upload_json(file: UploadFile = File(...), session_id: str = Form(...)):
    contents = await file.read()
    data = json.loads(contents)
    session_data[session_id] = data
    return JSONResponse(content={'status': 'success', 'data': data})


@app.post('/manual-input/')
async def manual_input(
    position: str = Form(...),
    age: int = Form(...),
    country: str = Form(...),
    city: str = Form(...),
    key_skills: str = Form(...),
    work_experience: str = Form(...),
    session_id: str = Form(...)
):
    data = {
        'position': position,
        'age': age,
        'country': country,
        'city': city,
        'key_skills': key_skills,
        'work_experience': work_experience
    }
    session_data[session_id] = data
    return JSONResponse(content={'status': 'success', 'data': data})


@app.post("/process-data/")
async def process_data(
    client_name: str = Form(...),
    expected_grade_salary: str = Form(...),
    session_id: str = Form(...)
):
    data = session_data.get(session_id, {})
    if not data:
        return JSONResponse(content={'status': 'error', 'message': 'No data found for this session.'})
    data.update({
        'client_name': client_name,
        'salary': expected_grade_salary,
    })
    df = pd.DataFrame([data])
    prediction = get_bert_prediction(df)
    results_dict = {
        'prediction': prediction,
        'resume_details': data
    }
    session_data[session_id] = results_dict
    return JSONResponse(content={'status': 'success', 'prediction': prediction, 'data': results_dict})


@app.get('/download-results/')
async def download_results(session_id: str):
    data = session_data.get(session_id, {})
    if not data:
        return JSONResponse(content={'status': 'error', 'message': 'No data found for this session.'})
    results_dict = {
        'prediction': data.get('prediction', 0.0),
        'resume_details': data.get('resume_details', {})
    }
    json_data = json.dumps(results_dict, ensure_ascii=False, indent=4).encode('utf-8')
    return StreamingResponse(
        iter([json_data]),
        media_type='application/json',
        headers={
            'Content-Disposition': 'attachment; filename=resume_analysis_result.json'
        }
    )
