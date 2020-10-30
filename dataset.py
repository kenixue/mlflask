import csv
import datetime
from functools import reduce


class Dataset:
    def __init__(self, file_path):
        # self.raw_results = []
        self.processed_results = []

        with open(file_path) as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                processed_result = {
                    'DEATH_EVENT': row['DEATH_EVENT'],
                    'age': float(row['age']),
                    'anaemia': float(row['anaemia']),
                    'creatinine_phosphokinase': float(row['creatinine_phosphokinase']),
                    'diabetes': float(row['diabetes']),
                    'ejection_fraction': float(row['ejection_fraction']),
                    'high_blood_pressure': float(row['high_blood_pressure']),
                    'platelets': float(row['platelets']),
                    'serum_creatinine': float(row['serum_creatinine']),
                    'serum_sodium': float(row['serum_sodium']),
                    'sex': float(row['sex']),
                    'smoking': float(row['smoking']),
                    'time': float(row['time']),
                }
                self.processed_results.append(processed_result)
        print(len(self.processed_results))


