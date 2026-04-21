import os
import re
import tgt
import chardet
import pandas as pd
import json
import yaml

OUTPUT_PATH       = os.path.join("data", "parsed", "words.csv")
USABLE_WORDS_PATH = os.path.join("data", "parsed", "usable_words.json")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

CORPUS_DIR    = params["corpus"]["dir"]
METADATA_PATH = params["corpus"]["metadata"]
NOISE_TOKENS  = set(params["parse"]["noise_tokens"])
MIN_SPEAKERS  = params["parse"]["min_speakers"]
MIN_RECORDINGS = params["parse"]["min_recordings"]

# metadata

meta = pd.read_csv(METADATA_PATH, sep=";")
meta["spk_lower"] = meta["spk"].str.lower()
speaker_info = meta.set_index("spk_lower")[["spk", "L1", "Gender"]].to_dict(orient="index")


def read_txt(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    encoding = chardet.detect(raw)["encoding"] or "utf-8"
    return raw.decode(encoding).strip()


def clean_word(w: str) -> str:
    w = w.strip().lower()
    w = re.sub(r"[^a-zàâäéèêëîïôùûüçœæ']", "", w)
    return w


# parse

rows: list[dict] = []

for speaker_folder in sorted(os.listdir(CORPUS_DIR)):
    speaker_path = os.path.join(CORPUS_DIR, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    spk_id = speaker_folder.lower()
    if spk_id not in speaker_info:
        print(f"[WARN] no metadata for speaker '{spk_id}', skipping")
        continue

    spk_original = speaker_info[spk_id]["spk"]
    l1           = speaker_info[spk_id]["L1"]
    gender       = speaker_info[spk_id]["Gender"]

    for filename in sorted(os.listdir(speaker_path)):
        if not filename.endswith(".TextGrid"):
            continue

        match = re.match(
            r"(?P<spk>[^_]+)_(?P<l1>[^_]+)_(?P<list>[^_]+)_(?P<sent_id>FRcorp\d+)\.TextGrid",
            filename,
            re.IGNORECASE,
        )
        if not match:
            print(f"[WARN] could not parse filename '{filename}', skipping")
            continue

        sent_id  = match.group("sent_id").upper()
        tg_path  = os.path.join(speaker_path, filename)
        wav_path = tg_path.replace(".TextGrid", ".wav")
        txt_path = tg_path.replace(".TextGrid", ".txt")

        if not os.path.isfile(wav_path):
            print(f"[WARN] no wav found for '{tg_path}', skipping")
            continue

        sentence_text = read_txt(txt_path)

        try:
            tg         = tgt.io.read_textgrid(tg_path)
            words_tier = tg.get_tier_by_name("words")
        except Exception as e:
            print(f"[WARN] skipping '{tg_path}': {e}")
            continue

        for interval in words_tier.intervals:
            word = clean_word(interval.text)
            if word in NOISE_TOKENS:
                continue

            rows.append({
                "speaker_id":     spk_id,
                "speaker_id_raw": spk_original,
                "l1_status":      l1,
                "gender":         gender,
                "sent_id":        sent_id,
                "sentence_text":  sentence_text,
                "word":           word,
                "onset":          interval.start_time,
                "offset":         interval.end_time,
                "duration_ms":    round((interval.end_time - interval.start_time) * 1000, 4),
                "wav_path":       wav_path,
            })


# build dataframe

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No word tokens extracted. Check CORPUS_DIR and metadata paths.")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# usable words: ≥2 speakers AND ≥2 recordings per speaker
# every row is one distinct recording of a word by a speaker,
# so we just count rows per (word, speaker_id).

recordings_per_speaker = df.groupby(["word", "speaker_id"]).size()

words_with_enough = (
    recordings_per_speaker[recordings_per_speaker >= MIN_RECORDINGS]
    .groupby("word")
    .size()
)
word_speakers = df.groupby("word")["speaker_id"].nunique()

usable_mask  = (words_with_enough >= MIN_SPEAKERS) & (word_speakers >= MIN_SPEAKERS)
usable_words = sorted(usable_mask[usable_mask].index.tolist())

with open(USABLE_WORDS_PATH, "w", encoding="utf-8") as f:
    json.dump(usable_words, f, ensure_ascii=False, indent=2)

# summary

print(f"Done: {len(df)} word tokens → {OUTPUT_PATH}")
print(f"  Speakers        : {df['speaker_id'].nunique()}")
print(f"  Unique words    : {df['word'].nunique()}")
print(f"  Unique sentences: {df['sentence_text'].nunique()}")
print(f"\nWords usable for analysis: {len(usable_words)}")
print(usable_words)