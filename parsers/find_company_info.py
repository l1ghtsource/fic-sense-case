import pandas as pd


def find_company_raing(company: str, rating_df: pd.DataFrame):
    company_info = rating_df[rating_df['Название компании'] == company]

    # Проверяем если в названии компании несколько слов и пытаемся спарсить их
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
                company_kind=row['kind'],  # К какой категории относится
                region=row['Регион'],     # В каком регионе базируется
                field=row['Отрасль'],   # К какой отрасли относится
                # Итоговый балл в своей категории
                score=f"{float(row['Итоговый балл'])} из {rating_df[rating_df['kind'] == row['kind']]['Итоговый балл'].max()}"
            )
            return_array.append(info)
        return return_array

    return 'Company not found in HH.ru 2023 rating.'
