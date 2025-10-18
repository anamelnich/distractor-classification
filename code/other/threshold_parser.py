import re
import pandas as pd

# load the text
with open("./../cnbiLoop/online_decoders/e17_thresholds_log.txt","r") as f:
    text = f.read()

# regex for each run block
pattern = re.compile(
    r"Run timestamp: 2025-09-(\d+) [^\n]+\n"
    r"Decoder ambivalence margin: ([0-9.]+)\n"
    r"DecoderR threshold: ([0-9.]+)\n"
    r"DecoderL threshold: ([0-9.]+)\n"
    r"DecoderN threshold: ([0-9.]+)",
    re.MULTILINE
)

rows = []
session_map = {"17":1, "19":2, "22":3, "24":4, "26":5}
run_counters = {k:0 for k in session_map.values()}

for m in pattern.finditer(text):
    day, margin, thrR, thrL, thrN = m.groups()
    session = session_map.get(day)
    if session is None:
        continue
    run_counters[session] += 1
    run = run_counters[session]
    rows.append([session, run,
                 float(margin),
                 float(thrR), float(thrL), float(thrN)])

df = pd.DataFrame(rows, columns=["Session","Run","Margin","ThresholdR","ThresholdL","ThresholdN"])

# save to CSV for MATLAB
df.to_csv("thresholds_table.csv", index=False)
print(df.head())
