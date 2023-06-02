import json
import pathlib
import urllib.request

from eda import *

# Вводные для удобства.
current_script_name = pathlib.Path(__file__).name
json_url = "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"

# Скачивает json c сайта, формирует датафрейм, сохраняет голову в imtermediate data/heads.
with urllib.request.urlopen(json_url) as url:
    data = json.load(url)
    original_df = pd.DataFrame(data)
    inspect_data(current_script_name, original_df, 'original_df')

