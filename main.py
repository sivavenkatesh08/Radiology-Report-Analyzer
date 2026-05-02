from src.nlp.summarizer import summarize_report
from src.vision.image_processor import highlight_region
from src.nlp.ner import extract_entities
import os
import cv2
report = """
There is a mild opacity observed in the right lung region, suggesting early infection. 
No pleural effusion detected. Heart size is normal.
"""
#summarizer
summary = summarize_report(report)

print("\n===== SUMMARY =====")
print(summary)

#ner
entities = extract_entities(report)

print("\n===== ENTITIES =====")
for e in entities:
    print(e)


#image processing
print("\nProcessing Image...")

output = highlight_region("D:\Radiology Report Analyzer\data\images\sample x-ray.jpg")

cv2.imshow("Highlighted X-ray", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

from src.nlp.ocr import extract_text_from_image

text = extract_text_from_image("D:\Radiology Report Analyzer\data\images\sample x-ray.jpg")
print(text)