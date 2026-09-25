import sys, os
import pandas as pd
sys.path.insert(0, "src")
from io_utils import read_source

os.makedirs("data_sample/test_tiny", exist_ok=True)

s1 = read_source("../../student_resource/dataset/test/test_source1.tsv").head(200)
s2 = read_source("../../student_resource/dataset/test/test_source2.tsv")
s3 = read_source("../../student_resource/dataset/test/test_source3.tsv")

# no ground truth on test side, so no "needed ids" step -- just random padding
s2_final = s2.sample(n=min(3000, len(s2)), random_state=0)
s3_final = s3.sample(n=min(3000, len(s3)), random_state=0)

s1.to_csv("data_sample/test_tiny/test_source1.tsv", sep="\t", index=False)
s2_final.to_csv("data_sample/test_tiny/test_source2.tsv", sep="\t", index=False)
s3_final.to_csv("data_sample/test_tiny/test_source3.tsv", sep="\t", index=False)

print("tiny test sample:", s1.shape, s2_final.shape, s3_final.shape)