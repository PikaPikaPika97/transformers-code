from pathlib import Path

import requests
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

BASE_URL = "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People's%20Daily"

files = {
    "train": "example.train",
    "validation": "example.dev",
    "test": "example.test",
}

label_names = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]

features = Features(
    {
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_names)),
    }
)


def download_file(split, filename, data_dir="peoples_daily_ner_raw"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/{filename}"
    out_path = data_dir / filename

    if not out_path.exists():
        r = requests.get(url)
        r.raise_for_status()
        out_path.write_text(r.text, encoding="utf-8")

    return out_path


def parse_bio_file(path):
    examples = []
    tokens = []
    tags = []
    guid = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if tokens:
                    examples.append(
                        {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": [label_names.index(tag) for tag in tags],
                        }
                    )
                    guid += 1
                    tokens = []
                    tags = []
                continue

            parts = line.split(" ")
            if len(parts) == 1:
                token, tag = parts[0], "O"
            else:
                token, tag = parts[0], parts[1]

            tokens.append(token)
            tags.append(tag)

    if tokens:
        examples.append(
            {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": [label_names.index(tag) for tag in tags],
            }
        )

    return examples


dataset_dict = {}

for split, filename in files.items():
    path = download_file(split, filename)
    examples = parse_bio_file(path)
    dataset_dict[split] = Dataset.from_list(examples, features=features)

ds = DatasetDict(dataset_dict)
ds.save_to_disk("my_peoples_daily_ner_dataset")
