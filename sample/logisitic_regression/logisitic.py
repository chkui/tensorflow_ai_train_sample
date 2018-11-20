import pandas as pd

admissions = pd.read_csv('admissions.csv').get_values()
sub = admissions[0]
print(sub[1])
print(admissions)