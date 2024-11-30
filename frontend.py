import streamlit as st
import json
import pandas as pd
from ml.bert_inference import get_bert_prediction
from parsers.hh_document_parser import parse_hh_pdf
from parsers.hh_link_parser import parse_hh_link

hide_decoration_bar_style = '''
    <style>
    header {visibility: hidden;}
    base="light"
    primaryColor="#0077ff"
    .reportview-container {
        background: green
    }
    .round {
        border-radius: 100px; /* Радиус скругления */
        border: 3px solid white; /* Параметры рамки */
    }
    </style>
'''

st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
st.markdown(
    '''
        <style>
        body {
            background-color: green;
        }
        </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True
)

st.markdown('''
    <style>
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        background-color: white;
    }
    .header img {
        width: 100%;
        height: auto;
    }
    body {
        padding-top: 100px;
    }
    </style>
    <div class="header">
        <img src="https://i.imgur.com/vYjM83q.png" alt="Header">
    </div>
''', unsafe_allow_html=True)


def main():
    st.title('Резюме.тч')

    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None
    if 'hh_link_data' not in st.session_state:
        st.session_state.hh_link_data = None
    if 'json_data' not in st.session_state:
        st.session_state.json_data = None
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = None

    tabs = st.tabs(['Загрузить PDF', 'Вставить ссылку на HH', 'Загрузить JSON', 'Ввести вручную'])

    with tabs[0]:
        st.header("Загрузить PDF")
        uploaded_file = st.file_uploader('Загрузите PDF резюме', type='pdf', key='pdf_uploader')
        if uploaded_file is not None:
            st.session_state.hh_link_data = None
            st.session_state.json_data = None
            st.session_state.manual_data = None

            with open('temp_resume.pdf', 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.pdf_data = parse_hh_pdf('temp_resume.pdf')
            st.success('Резюме успешно загружено и обработано')
        elif st.session_state.pdf_data:
            st.json(st.session_state.pdf_data)

    with tabs[1]:
        st.header('Вставить ссылку на HH')
        hh_link = st.text_input('Введите ссылку на резюме с HeadHunter', key="hh_link_input")
        if hh_link:
            st.session_state.pdf_data = None
            st.session_state.json_data = None
            st.session_state.manual_data = None

            if 'hh.ru/resume/' in hh_link:
                st.session_state.hh_link_data = parse_hh_link(hh_link)
                st.success('Резюме успешно обработано')
            else:
                st.session_state.hh_link_data = None
                st.error('Ссылка должна быть на HH резюме!')
        elif st.session_state.hh_link_data:
            st.json(st.session_state.hh_link_data)

    with tabs[2]:
        st.header('Загрузить JSON')
        uploaded_file = st.file_uploader('Загрузите JSON файл с данными', type='json', key='json_uploader')
        if uploaded_file is not None:
            st.session_state.pdf_data = None
            st.session_state.hh_link_data = None
            st.session_state.manual_data = None

            st.session_state.json_data = pd.read_json(uploaded_file).iloc[0].to_dict()
            st.success('JSON успешно загружен и обработан')
        elif st.session_state.json_data:
            st.json(st.session_state.json_data)

    with tabs[3]:
        st.header('Ввести вручную')
        st.session_state.manual_data = {
            'position': st.text_input('Должность', key='manual_position'),
            'age': st.number_input('Возраст', min_value=18, max_value=100, step=1, key='manual_age'),
            'country': st.text_input('Страна', key='manual_country'),
            'city': st.text_input('Город', key='manual_city'),
            'key_skills': st.text_area('Ключевые навыки', key='manual_skills'),
            'work_experience': st.text_area('Опыт работы', key='work_experience')
        }

    client_name = st.text_input('Название компании')
    expected_grade_salary = st.text_input('Ожидаемый грейд и зарплата')

    if st.button('Обработать'):
        if st.session_state.pdf_data:
            final_dict = st.session_state.pdf_data.copy()
        elif st.session_state.hh_link_data:
            final_dict = st.session_state.hh_link_data.copy()
        elif st.session_state.json_data:
            final_dict = st.session_state.json_data.copy()
        elif st.session_state.manual_data:
            final_dict = st.session_state.manual_data.copy()
        else:
            st.error('Пожалуйста, загрузите данные или введите их вручную!')
            return

        if client_name and expected_grade_salary:
            final_dict.update({
                'client_name': client_name,
                'salary': expected_grade_salary,
            })

            df = pd.DataFrame([final_dict])
            prediction = get_bert_prediction(df)

            st.metric('Вероятность соответствия', f"{prediction:.2f}")

            results_dict = {
                'prediction': prediction,
                'resume_details': final_dict
            }

            with open('resume_analysis_result.json', 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)

            with open('resume_analysis_result.json', 'rb') as file:
                st.download_button(
                    label='Скачать результаты',
                    data=file,
                    file_name='resume_analysis_result.json',
                    mime='application/json'
                )
        else:
            st.error('Пожалуйста, заполните все необходимые поля!')


if __name__ == '__main__':
    main()
