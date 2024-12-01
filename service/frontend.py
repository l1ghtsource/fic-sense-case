import streamlit as st
import requests
import uuid

BASE_URL = 'http://localhost:8000'

st.set_page_config(
    page_title='Резюме.тч',
    page_icon='📝',
    initial_sidebar_state='collapsed'
)

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

    if 'mode' not in st.session_state:
        st.session_state['mode'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None

    if st.session_state['mode'] is None:
        st.header('Выберите режим работы')

        mode = st.radio(
            'Режим',
            ['Сервис для пользователя', 'Генерация сабмита из JSON'],
            key='mode_selection'
        )

        model = st.radio(
            'Модель анализа',
            ['BERT (0.53 Roc-Auc)', 'Gemma-2-9b-it-4bit (0.61 Roc-Auc)'],
            key='model_selection'
        )

        st.warning(
            '⚠️ Внимание: Модель **Gemma-2-9b-it-4bit** недоступна из-за отсутствия GPU для инференса. '
            'Просьба использовать BERT.'
        )

        if st.button('Подтвердить выбор'):
            st.session_state['mode'] = mode
            st.session_state['model'] = model
            st.rerun()

    else:
        session_id = st.session_state.get('session_id', None)
        if not session_id:
            session_id = st.session_state['session_id'] = str(uuid.uuid4())

        if st.sidebar.button('Изменить режим'):
            st.session_state['mode'] = None
            st.session_state['model'] = None
            st.rerun()

        st.sidebar.info(f"Режим: {st.session_state['mode']}")
        st.sidebar.info(f"Модель: {st.session_state['model']}")

        if st.session_state['mode'] == 'Сервис для пользователя':
            tabs = st.tabs(['Загрузить PDF', 'Вставить ссылку на HH', 'Ввести вручную'])

            with tabs[0]:
                st.header('Загрузить PDF')
                uploaded_file = st.file_uploader('Загрузите PDF резюме', type='pdf', key='pdf_uploader')
                if uploaded_file is not None:
                    files = {'file': uploaded_file}
                    data = {'session_id': session_id}
                    response = requests.post(f'{BASE_URL}/upload-pdf/', files=files, data=data)
                    if response.status_code == 200:
                        st.success('Резюме успешно загружено и обработано')
                        st.session_state['data'] = response.json()['data']
                        st.json(st.session_state['data'], expanded=False)
                    else:
                        st.error('Ошибка при загрузке PDF')

            with tabs[1]:
                st.header('Вставить ссылку на HH')
                hh_link = st.text_input('Введите ссылку на резюме с HeadHunter', key='hh_link_input')
                if hh_link:
                    data = {'link': hh_link, 'session_id': session_id}
                    response = requests.post(f'{BASE_URL}/process-hh-link/', data=data)
                    if response.status_code == 200:
                        st.success('Резюме успешно обработано')
                        st.session_state['data'] = response.json()['data']
                        st.json(st.session_state['data'], expanded=False)
                    else:
                        st.error('Ошибка при обработке ссылки HH')

            with tabs[2]:
                st.header('Ввести вручную')
                position = st.text_input('Должность', key='manual_position')
                age = st.number_input('Возраст', min_value=18, max_value=100, step=1, key='manual_age')
                city = st.text_input('Город', key='manual_city')
                key_skills = st.text_area('Ключевые навыки', key='manual_skills')
                work_experience = st.text_area('Опыт работы', key='work_experience')
                if st.button('Сохранить ввод'):
                    data = {
                        'position': position,
                        'age': age,
                        'city': city,
                        'key_skills': key_skills,
                        'work_experience': work_experience
                    }
                    response = requests.post(f'{BASE_URL}/manual-input/', data={**data, 'session_id': session_id})
                    if response.status_code == 200:
                        st.success('Данные успешно сохранены')
                        st.session_state['data'] = response.json()['data']
                    else:
                        st.error('Ошибка при сохранении данных')

            client_name = st.text_input('Название компании')
            expected_grade_salary = st.text_input('Ожидаемый грейд и зарплата')

            if st.button('Обработать'):
                if not st.session_state.get('data'):
                    st.error('Пожалуйста, загрузите данные!')
                elif not client_name or not expected_grade_salary:
                    st.error('Пожалуйста, заполните все необходимые поля!')
                else:
                    data = {
                        'client_name': client_name,
                        'expected_grade_salary': expected_grade_salary,
                        'session_id': session_id
                    }

                    model_endpoint = 'process-data-bert' if st.session_state['model'] == 'BERT (0.53 Roc-Auc)' else 'process-data-gemma'

                    response = requests.post(f'{BASE_URL}/{model_endpoint}/', data=data)
                    if response.status_code == 200:
                        prediction = response.json()['prediction']
                        if len(prediction) == 1:
                            prediction = prediction[0]
                        st.metric('Вероятность соответствия', f"{prediction:.5f}")

                        download_url = f'{BASE_URL}/download-results/?session_id={session_id}'

                        file_response = requests.get(download_url)
                        if file_response.status_code == 200:
                            st.download_button(
                                label='Скачать результаты',
                                data=file_response.content,
                                file_name='resume_analysis_result.json',
                                mime='application/json'
                            )
                        else:
                            st.error('Ошибка при скачивании результатов.')
                    else:
                        st.error('Ошибка при обработке данных')

        elif st.session_state['mode'] == 'Генерация сабмита из JSON':
            st.header('Загрузить JSON для генерации сабмита')
            uploaded_file = st.file_uploader('Загрузите JSON файл с данными', type='json', key='json_uploader')
            if uploaded_file is not None:
                files = {'file': uploaded_file}
                data = {'session_id': session_id}
                response = requests.post(f'{BASE_URL}/upload-json/', files=files, data=data)
                if response.status_code == 200:
                    st.success('JSON успешно загружен и обработан')
                    res = response.json()['data']
                    st.session_state['data'] = res
                    st.json(st.session_state['data'], expanded=False)
                    client_name = res['client_name']
                    expected_grade_salary = res['salary']
                    print(client_name, expected_grade_salary)
                else:
                    st.error('Ошибка при загрузке JSON')

            if st.button('Обработать'):
                if not st.session_state.get('data'):
                    st.error('Пожалуйста, загрузите данные!')
                else:
                    data = {
                        'client_name': client_name,
                        'expected_grade_salary': expected_grade_salary,
                        'session_id': session_id
                    }

                    model_endpoint = 'process-data-bert-json' if st.session_state[
                        'model'] == 'BERT (0.53 Roc-Auc)' else 'process-data-gemma-json'

                    response = requests.post(f'{BASE_URL}/{model_endpoint}/', data=data)
                    if response.status_code == 200:
                        prediction = response.json()['prediction']
                        if len(prediction) == 1:
                            prediction = prediction[0]
                            st.metric('Вероятность соответствия', f'{prediction:.5f}')
                        else:
                            with st.expander('Вероятности соответствия:'):
                                st.table(prediction)

                        download_url = f'{BASE_URL}/download-results/?session_id={session_id}'

                        file_response = requests.get(download_url)
                        if file_response.status_code == 200:
                            st.download_button(
                                label='Скачать результаты',
                                data=file_response.content,
                                file_name='resume_analysis_result.json',
                                mime='application/json'
                            )
                        else:
                            st.error('Ошибка при скачивании результатов.')
                    else:
                        st.error('Ошибка при обработке данных')


if __name__ == '__main__':
    main()
