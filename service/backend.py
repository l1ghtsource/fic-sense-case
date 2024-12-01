import pandas as pd
import json
import torch
import ast

from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from peft import PeftModel
from dataclasses import dataclass

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from ml.bert_inference import RuBERTInferenceModel
from ml.gemma_inference import get_gemma_prediction
from parsers.hh_document_parser import parse_hh_pdf
from parsers.hh_link_parser import parse_hh_link

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@dataclass
class BertConfig:
    checkpoint: str = 'lightsource/fic-rubert-tiny-2-chckpnt800'
    hdim: int = 312
    num_labels: int = 2
    max_length: int = 512
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


bert_config = BertConfig()
bert_model = RuBERTInferenceModel(bert_config)


# @dataclass
# class GemmaConfig:
#     gemma_dir = 'unsloth/gemma-2-9b-it-bnb-4bit'
#     lora_dir = ''
#     max_length = 2048
#     batch_size = 4
#     device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# gemma_cfg = GemmaConfig()

# gemma_tokenizer, gemma_model = None, None


# def load_tokenizer_and_model(cfg):
#     global gemma_tokenizer, gemma_model
#     gemma_tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
#     gemma_tokenizer.add_eos_token = True
#     gemma_tokenizer.padding_side = 'right'

#     gemma_model = Gemma2ForSequenceClassification.from_pretrained(
#         cfg.gemma_dir,
#         device_map=cfg.device,
#         use_cache=False
#     )

#     gemma_model = PeftModel.from_pretrained(gemma_model, cfg.lora_dir)
#     gemma_model.eval()


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
    city: str = Form(...),
    key_skills: str = Form(...),
    work_experience: str = Form(...),
    session_id: str = Form(...)
):
    data = {
        'position': position,
        'age': age,
        'city': city,
        'key_skills': key_skills,
        'work_experience': work_experience
    }
    session_data[session_id] = data
    return JSONResponse(content={'status': 'success', 'data': data})


@app.post('/process-data-bert/')
async def process_data_bert(
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
    df.to_csv('ahahhaha.csv', index=False)
    if 'country' in df.columns.tolist():
        df = df.drop(columns=['country'], axis=1)
    if 'grade_proof' in df.columns.tolist():
        df = df.drop(columns=['grade_proof'], axis=1)

    prediction = bert_model.predict_dataframe(df)
    results_dict = {
        'prediction': prediction,
        'resume_details': data
    }
    session_data[session_id] = results_dict
    return JSONResponse(content={'status': 'success', 'prediction': prediction, 'data': results_dict})


@app.post('/process-data-bert-json/')
async def process_data_bert_json(
    client_name: str = Form(...),
    expected_grade_salary: str = Form(...),
    session_id: str = Form(...)
):
    data = session_data.get(session_id, {})
    if not data:
        return JSONResponse(content={'status': 'error', 'message': 'No data found for this session.'})

    df = pd.DataFrame([data])

    if 'country' in df.columns.tolist():
        df = df.drop(columns=['country'], axis=1)
    if 'grade_proof' in df.columns.tolist():
        df = df.drop(columns=['grade_proof'], axis=1)

    def safe_parse(value):
        try:
            if isinstance(value, str):
                return ast.literal_eval(value)
            return value
        except (ValueError, SyntaxError):
            return {}

    for col in df.columns:
        df[col] = df[col].apply(safe_parse)

    rows = []
    for _, row in df.iterrows():
        max_length = max(len(row[col]) if isinstance(row[col], dict) else 0 for col in df.columns)
        for i in range(max_length):
            new_row = {col: row[col].get(str(i), None) if isinstance(row[col], dict) else None for col in df.columns}
            rows.append(new_row)

    df = pd.DataFrame(rows)

    prediction = bert_model.predict_dataframe(df)
    results_dict = {
        'prediction': prediction,
        'resume_details': data
    }
    session_data[session_id] = results_dict
    return JSONResponse(content={'status': 'success', 'prediction': prediction, 'data': results_dict})


# @app.post('/process-data-gemma/')
# async def process_data_gemma(
#     client_name: str = Form(...),
#     expected_grade_salary: str = Form(...),
#     session_id: str = Form(...)
# ):
#     data = session_data.get(session_id, {})
#     if not data:
#         return JSONResponse(content={'status': 'error', 'message': 'No data found for this session.'})
#     data.update({
#         'client_name': client_name,
#         'salary': expected_grade_salary,
#     })
#     df = pd.DataFrame([data])
#     class_0, class_1 = get_gemma_prediction(
#         df, gemma_model, gemma_tokenizer, batch_size=gemma_cfg.batch_size, device=gemma_cfg.device,
#         max_length=gemma_cfg.max_length)
#     results_dict = {
#         'prediction': class_1,
#         'resume_details': data
#     }
#     session_data[session_id] = results_dict
#     return JSONResponse(content={'status': 'success', 'prediction': class_1, 'data': results_dict})


# @app.post('/process-data-gemma-json/')
# async def process_data_gemma_json(
#     client_name: str = Form(...),
#     expected_grade_salary: str = Form(...),
#     session_id: str = Form(...)
# ):
#     data = session_data.get(session_id, {})
#     if not data:
#         return JSONResponse(content={'status': 'error', 'message': 'No data found for this session.'})

#     df = pd.DataFrame([data])

#     if 'country' in df.columns.tolist():
#         df = df.drop(columns=['country'], axis=1)
#     if 'grade_proof' in df.columns.tolist():
#         df = df.drop(columns=['grade_proof'], axis=1)

#     def safe_parse(value):
#         try:
#             if isinstance(value, str):
#                 return ast.literal_eval(value)
#             return value
#         except (ValueError, SyntaxError):
#             return {}

#     for col in df.columns:
#         df[col] = df[col].apply(safe_parse)

#     rows = []
#     for _, row in df.iterrows():
#         max_length = max(len(row[col]) if isinstance(row[col], dict) else 0 for col in df.columns)
#         for i in range(max_length):
#             new_row = {col: row[col].get(str(i), None) if isinstance(row[col], dict) else None for col in df.columns}
#             rows.append(new_row)

#     df = pd.DataFrame(rows)

#     prediction = bert_model.predict_dataframe(df)
#     results_dict = {
#         'prediction': prediction,
#         'resume_details': data
#     }
#     session_data[session_id] = results_dict
#     return JSONResponse(content={'status': 'success', 'prediction': prediction, 'data': results_dict})


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
