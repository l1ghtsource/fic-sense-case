import os
from typing import Optional
import PyPDF2
from parsers.hh_link_parser import get_age_from_russian_date


def extract_text_from_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Файл не найден: {file_path}')

    file_extension = os.path.splitext(file_path)[1].lower()

    extractors = {
        '.pdf': extract_pdf_text,
    }

    extractor = extractors.get(file_extension)

    if not extractor:
        raise ValueError(f'Неподдерживаемый формат файла: {file_extension}')

    return extractor(file_path)


def extract_pdf_text(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()


def get_position(text):
    try:
        res = text.split('Специализации:\n')[1].split('Занятость')[0].strip()
    except:
        res = ''
    return res


def get_age(text):
    try:
        res = text.split('родился')[1].split('\n')[0].strip()
    except:
        res = ''
    return res


def get_city(text):
    try:
        res = text.split('Проживает:')[1].split('\n')[0].strip()
    except:
        res = ''
    return res


def get_exp(text):
    try:
        res = text.split('Опыт работы')[1].split('Образование')[0].strip()
    except:
        res = ''
    return res


def get_skills(text):
    try:
        res = text.split('Навыки')[2].split('Дополнительная информация')[0].strip()
    except:
        res = ''
    return res


def parse_hh_pdf(file_path):
    extracted_text = extract_text_from_file(file_path)

    parsed_data = {
        'position': get_position(extracted_text),
        'age': get_age_from_russian_date(get_age(extracted_text)),
        'city': get_city(extracted_text),
        'key_skills': get_skills(extracted_text),
        'client_name': '',
        'salary': '',
        'work_experience': get_exp(extracted_text)
    }

    return parsed_data
