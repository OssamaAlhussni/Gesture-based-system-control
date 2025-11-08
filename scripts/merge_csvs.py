import csv

files = [
    "data/palm_embeddings.csv",
    "data/fist_embeddings.csv",
    "data/thumbsup_embeddings.csv",
    "data/indexpoint_embeddings.csv",
    "data/peace_embeddings.csv"
]

rows = []
header = None

for f in files:
    with open(f, newline="") as infile:
        reader = csv.reader(infile)
        h = next(reader)
        if header is None:
            header = h
        for r in reader:
            rows.append(r)

with open("data/gestures_all.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(header)
    writer.writerows(rows)

print("Merged", len(rows), "samples")
