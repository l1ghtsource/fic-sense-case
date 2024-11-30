from parse_hh_data import download, parse
from datetime import datetime


def parse_russian_date(date_string):
    months = {
        'января': 1,
        'февраля': 2,
        'марта': 3,
        'апреля': 4,
        'мая': 5,
        'июня': 6,
        'июля': 7,
        'августа': 8,
        'сентября': 9,
        'октября': 10,
        'ноября': 11,
        'декабря': 12
    }

    date_string = date_string.replace('\xa0', ' ')

    parts = date_string.split()

    try:
        day = int(parts[0])
        month = months[parts[1]]
        year = int(parts[2])

        return datetime(year, month, day)
    except (ValueError, KeyError, IndexError):
        raise ValueError('Не удалось распознать дату')


def calculate_age(birth_date):
    today = datetime.now()
    age = today.year - birth_date.year

    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


def get_age_from_russian_date(date_string):
    birth_date = parse_russian_date(date_string)
    return calculate_age(birth_date)


def parse_work_experience(experience_list):
    if not experience_list:
        return ''

    descriptions = []
    for entry in experience_list:
        start_date = entry.get('start', 'неизвестно')
        end_date = entry.get('end', 'неизвестно')
        description = entry.get('description', '').replace('\n', ' ').replace('\r', ' ').strip()
        descriptions.append(f"{start_date} - {end_date}: {description}")

    combined_description = ' '.join(descriptions).strip()
    return combined_description


# на вход ссылка в виде https://hh.ru/resume/43a999b3ff0d5e49570039ed1f6a526c304250, на выходе df с заполненными полями в формате train
def parse_hh_link(link):
    hh_id = link.split('/')[-1]
    try:
        resume = download.resume(hh_id)
        resume = parse.resume(resume)

        parsed_data = {
            'position': resume.get('title', ''),
            'age': get_age_from_russian_date(resume.get('birth_date')),
            'country': '',
            'city': resume.get('area', ''),
            'key_skills': ', '.join([skill['name'] for skill in resume.get('skill_set', [])]),
            'client_name': '',
            'grade_proof': resume.get('education_level', ''),
            'salary': resume.get('salary', {}).get('amount', ''),
            'work_experience': parse_work_experience(resume.get('experience', []))
        }
    except:
        parsed_data = {
            'position': '',
            'age': '',
            'country': '',
            'city': '',
            'key_skills': '',
            'client_name': '',
            'grade_proof': '',
            'salary': '',
            'work_experience': ''
        }

    return parsed_data
