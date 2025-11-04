import re
import os, json
from collections import defaultdict
import pandas as pd
from openai import OpenAI

def extract_adjectives(answer, noun):
    # Capture sequences like "a sliced apple" or "a partially sliced apple"
    # Allow words connected by spaces or hyphens before the noun
    pattern = rf"\b(?:a|an)\s+([a-zA-Z\- ]+?)\s+{re.escape(noun)}\b"
    matches = re.findall(pattern, answer)
    # Clean up extra spaces
    return [m.strip() for m in matches]

image_directory = "openAI_images"
model_name = "gpt-5-mini"
system_message = "You analyze two images of objects and answer concisely."
prompt_template = "What is the most natural way to describe the two images and their states using the format 'a/an ADJECTIVE {noun}'? Please list three full phrases (in order of naturalness) for each image, for a total of six phrases (three per image)"
dry_run = False   # set to False to actually call the API

client = OpenAI()

results = []

grouped_images = defaultdict(list)
for filename in os.listdir(image_directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")) and "_" in filename:
        noun = filename.split("_", 1)[0]
        grouped_images[noun].append(filename)

# check whether each has two images
assert all(len(files) == 2 for files in grouped_images.values())

for noun, files in sorted(grouped_images.items()):
    prompt_text = prompt_template.format(noun=noun)
    image_paths = [os.path.join(image_directory, f) for f in sorted(files)]

    if dry_run:
        print(f"\nPrompt: {prompt_text}")
        print("Images:")
        for path in image_paths:
            print("  ", path)
        continue

    uploaded_ids = [client.files.create(file=open(p, "rb"), purpose="vision").id for p in image_paths]
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "file_id": uploaded_ids[0]},
                {"type": "input_image", "file_id": uploaded_ids[1]},
            ]},
        ],
    )

    
    answer_text = getattr(response, "output_text", None)
    print(noun)
    print(answer_text)
    print(" ")
    results.append({
        "noun": noun,
        "files": files,
        "prompt": prompt_text,
        "answer": answer_text
    })

results_dataframe = pd.DataFrame(results)
results_dataframe['adjectives'] = results_dataframe.apply(lambda row: extract_adjectives(row['answer'], row['noun']), axis=1)
results_dataframe['state_A_adj'] = results_dataframe['adjectives'].apply(lambda x: x[0:3])
results_dataframe['state_B_adj'] = results_dataframe['adjectives'].apply(lambda x: x[3:6])
results_dataframe.to_csv("openai_natural_adjective.csv", index=False)
