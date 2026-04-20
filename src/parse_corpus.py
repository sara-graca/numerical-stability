import os
import re
import tgt
import chardet
import pandas as pd
from collections import defaultdict

CORPUS_DIR    = os.path.join("data", "raw", "ru-fr_interference", "2", "wav_et_textgrids", "FRcorp_textgrids_only")
METADATA_PATH = os.path.join("data", "raw", "ru-fr_interference", "2", "metadata_RUFR.csv")
OUTPUT_PATH   = os.path.join("data", "parsed", "words.csv")

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
    # keep only letters and apostrophes (drops digits, underscores, punctuation)
    w = re.sub(r"[^a-zàâäéèêëîïôùûüçœæ']", "", w)
    return w

# parse
rows: list[dict] = []
# key: (spk_id, sent_id) → count of recordings seen so far
repetition_counter: dict[tuple[str, str], int] = defaultdict(int)

NOISE_TOKENS = {"sil", "sp", "spn", "unk", "<unk>", ""}

for speaker_folder in sorted(os.listdir(CORPUS_DIR)):
    speaker_path = os.path.join(CORPUS_DIR, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    spk_id = speaker_folder.lower()
    if spk_id not in speaker_info:
        print(f"[WARN] no metadata for speaker '{spk_id}', skipping")
        continue

    spk_original = speaker_info[spk_id]["spk"]
    l1 = speaker_info[spk_id]["L1"]
    gender = speaker_info[spk_id]["Gender"]

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

        # Use sent_id from the filename as the stable key (never missing)
        sent_id  = match.group("sent_id").upper()
        tg_path  = os.path.join(speaker_path, filename)
        wav_path = tg_path.replace(".TextGrid", ".wav")
        txt_path = tg_path.replace(".TextGrid", ".txt")

        # Check wav exists before going further
        if not os.path.isfile(wav_path):
            print(f"[WARN] no wav found for '{tg_path}', skipping")
            continue

        sentence_text = read_txt(txt_path)

        # Repetition index: incremented per (speaker, sent_id) pair
        key = (spk_id, sentence_text)
        repetition_counter[key] += 1
        rep_idx = repetition_counter[key]

        # Load TextGrid
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
                "speaker_id":    spk_id,
                "speaker_id_raw": spk_original,
                "l1_status":     l1,
                "gender":        gender,
                "sent_id":       sent_id,
                "sentence_text": sentence_text,
                "repetition":    rep_idx,
                "word":          word,
                "onset":         interval.start_time,
                "offset":        interval.end_time,
                "duration_ms":   round((interval.end_time - interval.start_time) * 1000, 4),
                "wav_path":      wav_path,
            })


# build df and save

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No word tokens were extracted. Check CORPUS_DIR and metadata paths.")

# Numeric sentence index (stable, sorted)
sent_ids_sorted = sorted(df["sent_id"].unique())
df["sentence_index"] = df["sent_id"].map({s: i + 1 for i, s in enumerate(sent_ids_sorted)})

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# summary

print(f"Done: {len(df)} word tokens → {OUTPUT_PATH}")
print(f"  Speakers  : {df['speaker_id'].nunique()}")
print(f"  Unique words: {df['word'].nunique()}")
print(f"  Unique sentences : {df['sentence_text'].nunique()}")
print(
    f"  Avg repetitions per speaker-sentence: "
    f"{df.groupby(['speaker_id', 'sentence_text'])['repetition'].max().mean():.2f}"
)

# Words usable for distance analysis (≥2 speakers, ≥2 reps per speaker)

word_speakers = df.groupby("word")["speaker_id"].nunique()

word_counts = df["word"].value_counts()

usable_words = list(word_speakers[word_speakers >= 2].index)

print(f"\nWords usable for analysis: {len(usable_words)}")

print(usable_words)