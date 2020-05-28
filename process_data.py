import numpy as np
import pandas as pd
import csv

datafile = 'data/essays.csv'

with open(datafile, "rt") as csvf:
    csvreader=csv.reader(csvf, delimiter=',',quotechar='"')
    first_line=True
    df = pd.DataFrame()
    for line in csvreader:
        if first_line:
            first_line=False
            continue
        
        essay = line[1]    

        tmp = pd.DataFrame({"user": line[0],
                "essay":essay,
                "cEXT":1 if line[2].lower()=='y' else 0,
                "cNEU":1 if line[3].lower()=='y' else 0,
                "cAGR":1 if line[4].lower()=='y' else 0,
                "cCON":1 if line[5].lower()=='y' else 0,
                "cOPN":1 if line[6].lower()=='y' else 0}, index=[0])
        
        df.append(tmp)

        
print(df.columns)
print(df.info)
print(df.head(10))