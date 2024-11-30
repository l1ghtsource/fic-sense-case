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

    input_method = st.radio(
        'Выберите способ загрузки резюме',
        ["Загрузить PDF", "Вставить ссылку на HH", "Загрузить JSON", "Ввести вручную"]
    )

    start_dict = {}

    if input_method == "Загрузить PDF":
        uploaded_file = st.file_uploader("Загрузите PDF резюме", type="pdf")
        if uploaded_file is not None:
            with open("temp_resume.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            start_dict = parse_hh_pdf("temp_resume.pdf")
    elif input_method == "Вставить ссылку на HH":
        hh_link = st.text_input("Введите ссылку на резюме с HeadHunter")
        if hh_link:
            if 'hh.ru/resume/' in hh_link:
                start_dict = parse_hh_link(hh_link)
            else:
                start_dict = None
                st.error("Ссылка должна быть на HH резюме!")
    elif input_method == "Загрузить JSON":
        uploaded_file = st.file_uploader("Загрузите JSON файл с данными", type="json")
        if uploaded_file is not None:
            start_dict = pd.read_json(uploaded_file).iloc[0].to_dict()
    elif input_method == "Ввести вручную":
        start_dict = {
            "position": st.text_input("Должность"),
            "age": st.number_input("Возраст", min_value=18, max_value=100, step=1),
            "country": st.text_input("Страна"),
            "city": st.text_input("Город"),
            "key_skills": st.text_area("Ключевые навыки"),
            "work_experience": st.number_input("Опыт работы (лет)", min_value=0, max_value=50, step=1)
        }

    client_name = st.text_input("Название компании")
    expected_grade_salary = st.text_input("Ожидаемый грейд и зарплата")

    if st.button("Обработать"):
        if start_dict and client_name and expected_grade_salary:
            final_dict = start_dict.copy()
            final_dict.update({
                "client_name": client_name,
                "salary": expected_grade_salary,
            })

            df = pd.DataFrame([final_dict])
            prediction = get_bert_prediction(df)

            st.metric("Вероятность соответствия", f"{prediction:.2f}")

            results_dict = {
                "prediction": prediction,
                "resume_details": final_dict
            }

            with open('resume_analysis_result.json', 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)

            with open('resume_analysis_result.json', 'rb') as file:
                st.download_button(
                    label="Скачать результаты",
                    data=file,
                    file_name='resume_analysis_result.json',
                    mime='application/json'
                )
        else:
            st.error("Пожалуйста, заполните все необходимые поля!")


if __name__ == '__main__':
    main()
