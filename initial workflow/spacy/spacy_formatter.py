import pandas as pd
import json
def df_maker():
    df = pd.read_csv("../Project_5/data/harvey/harvey_hoax.csv")
    df2 = pd.read_json("./conspiracy01_1000.jsonl", orient = "records", lines = True)
    df3 = pd.concat([df[['text']],df2])
    df3= df3.sample(frac=1)
    return df3

if __name__ == '__main__':

    df = df_maker()
    print(df['text'])
    df.to_json("./shuffled_harvey.jsonl", orient = "records", lines = True)
