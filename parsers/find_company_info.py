import pandas as pd


def find_company_rating(company: str, rating_df: pd.DataFrame):
    rating_df['Название компании'] = rating_df['Название компании'].apply(lambda x: x.lower())
    company = company.lower()

    company_info = rating_df[rating_df['Название компании'] == company]

    # проверяем если в названии компании несколько слов и пытаемся спарсить их
    if company_info.shape[0] == 0 and len(company.split()) > 1:
        company_info = pd.DataFrame()
        for word in company.split():
            company_info = pd.concat([company_info, rating_df[rating_df['Название компании'] == word]])

    if company_info.shape[0] > 0:
        return_array = []
        for _, row in company_info.iterrows():
            info = dict(
                company=company,
                place=f"{row['Место']} место из {rating_df[rating_df['kind'] == row['kind']]['Место'].max()}",
                company_kind=row['kind'],  # к какой категории относится
                region=row['Регион'],  # в каком регионе базируется
                field=row['Отрасль'],  # к какой отрасли относится
                # итоговый балл в своей категории
                score=f"{float(row['Итоговый балл'])} из {rating_df[rating_df['kind'] == row['kind']]['Итоговый балл'].max()}"
            )
            return_array.append(info)
        return return_array

    return 'Company not found in HH.ru 2023 rating.'
