# JSON failidest Exceli koostamine
# Raivo Kasepuu
# B710710
# 13.04.2023


import os
import json
import pandas as pd
import time


# JSON tulemuste kaust
RESULTS_JSON_FOLDER = "/Users/raivo/Documents/Magister_files/augJSON/resJSONs/"

# Loend JSON-failide nimedest kaustas
file_list = os.listdir(RESULTS_JSON_FOLDER)

# Loend, mis hoiab andmeid iga faili kohta
data_list = []

# Loe iga faili sisu ja lisage see loendisse
for file_name in file_list:
    if file_name.endswith('.json'):
        file_path = os.path.join(RESULTS_JSON_FOLDER, file_name)
        with open(file_path) as f:
            data = json.load(f)
            data_list.append(data)

# Looge andmeid käsitlev DataFrame
df = pd.DataFrame(data_list)
timestamp = int(time.time())
# Salvesta DataFrame Exceli failina
filename = RESULTS_JSON_FOLDER + "Results_excel_" + str(timestamp) + ".xlsx"
df.to_excel(filename, index=False)

