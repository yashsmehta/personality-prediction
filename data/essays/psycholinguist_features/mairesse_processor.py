import pandas as pd
import numpy as np

if __name__ == "__main__":
    mairesse = pd.read_csv("essays_mairesse.csv", header=None)
    columns = pd.read_csv("mairesse_attributes.csv", header=None)
    labels = np.transpose(columns.values.tolist())[0].tolist()
    labels.remove('BROWN-FREQ numeric')
    labels.remove('K-F-FREQ numeric')
    labels.remove('K-F-NCATS numeric')
    labels.remove('K-F-NSAMP numeric')
    labels.remove('T-L-FREQ numeric')
    mairesse.columns = labels
    mairesse = mairesse[mairesse.columns[:-5]]
    mairesse.to_csv("essays_mairesse_labelled.csv", index=False)
    print("done")
