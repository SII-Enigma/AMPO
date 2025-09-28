from datasets import load_dataset
import pandas as pd

dataset = load_dataset("SII-Enigma/SFT-4LongCoTs-32k", split="train")

print(dataset[0])

ret_dict = []
for item in dataset:
    ret_dict.append(item)

train_df = pd.DataFrame(ret_dict)
train_df.to_json("../data/openr1_math_4longcots_sft.json")