from transformers import pipeline

ner_pipeline = pipeline(
    "token-classification",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

def extract_entities(text):
    entities = ner_pipeline(text)

    cleaned = []
    for e in entities:
        word = e["word"].replace("##", "")
        cleaned.append({
            "entity": e["entity_group"],
            "word": word,
            "confidence": float(e["score"])
        })
    return cleaned