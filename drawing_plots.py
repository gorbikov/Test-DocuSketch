import json
import urllib.request
from pathlib import Path

import pandas as pd

class DrawingPlots:

    original_df: pd.DataFrame

    def draw_plots(self, json_url: str):
        # Скачивает json c сайта, формирует датафрейм.
        with urllib.request.urlopen(json_url) as url:
            data = json.load(url)
            original_df = pd.DataFrame(data)
            save_to_tmp('drawing_plots.py', original_df, 'original_df')
            print(original_df[original_df['gt_corners'] != original_df['rb_corners']])
            inspect_data('drawing_plots.py', original_df, 'original_df')
            count_unique(original_df, 'original_df')


# Функции-помощники.
def show_separator(*title_texts: str, size='small'):
    """Выводит разделитель в виде штриховой линии."""
    match size:
        case "small":
            print()
            for title_text in title_texts:
                print(title_text)
            print("---------------------------------------------------------------------")
        case "large":
            print()
            print()
            for title_text in title_texts:
                print(title_text)
            print(
                "===================================================================================================")

def save_to_tmp(current_script_name: str, df: pd.DataFrame, filename: str):
    """Сохраняет датафрейм в виде csv в папку intermediate data/tmp/."""
    show_separator("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("intermediate data/tmp/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))


def inspect_data(current_script_name: str, df: pd.DataFrame, filename: str):
    """Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data/heads/."""
    show_separator("Информация по " + filename)
    # Выводит инфо.
    df.info()
    # Сохраняет голову в файл.
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_head.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.head().to_csv(filepath)
    # Сохраняет статистику в файл.
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_describe.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.describe().to_csv(filepath)


def search_duplicates(df: pd.DataFrame, df_name: str):
    """Выводит количество дубликатов в датафрейме."""
    show_separator("Поиск дубликатов в " + df_name)
    print("Количество дубликатов в " + df_name + ":")
    print(df[df.duplicated()].shape[0])


def delete_duplicates(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Удаляет дубликаты в датафрейме."""
    show_separator("Удаление дубликатов в " + df_name)
    print("Размер " + df_name + " до удаления дубликатов: " + str(df.shape))
    df = df.drop_duplicates()
    print("Размер " + df_name + " после удаления дубликатов: " + str(df.shape))
    return df


def show_nans(df: pd.DataFrame, df_name: str):
    """Выводит количество пустых клеток в датафрейме."""
    show_separator("Поиск пустых клеток в " + df_name)
    print("Количество строк с пустыми клетками в " + df_name + ":")
    print(df[df.isnull().any(axis=1)].shape[0])
    print("Количество столбцов с пустыми клетками в " + df_name + ":")
    print(df.loc[:, df.isnull().any()].columns.size)


def count_unique_for_object_type(df: pd.DataFrame, df_name: str):
    """Выбирает столбцы типа object, выводит количество уникальных записей в каждом таком столбце."""
    show_separator("Уникальные значения в столбцах типа object в " + df_name)
    dummy_counter = 0
    for col in df.columns:
        if df[col].dtypes == object:
            dummy_counter += len(df[col].unique())
            print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))

    print('Dummy columns: ' + str(dummy_counter))


def count_unique(df: pd.DataFrame, df_name: str):
    """Выводит количество уникальных записей в каждом столбце."""
    show_separator("Уникальные значения в столбцах в " + df_name)
    for col in df.columns:
        print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))