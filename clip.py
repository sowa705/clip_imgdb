from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import sqlite3
import os
import hashlib
import json
import sys

def import_image(image_path):
    if not os.path.isfile(image_path):
        return
    try:
        image=Image.open(image_path)
    except:
        print("Error opening image")
        return
    # we read the image into memory
    image_tensor = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to(device)
    embedding = model.get_image_features(image_tensor)

    embedding_json = json.dumps(embedding.tolist())
    # we hash the image using sha256
    hash = hashlib.sha256(image.tobytes()).hexdigest()

    db.execute("INSERT INTO images (hash, paths, embedding) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            (hash, image_path, embedding_json))

    db.commit()

def search_image(prompt):
    # convert the prompt to a tensor
    text = tokenizer(prompt, return_tensors="pt").to(device)
    # get the embedding of the prompt
    text_features = model.get_text_features(**text)
    # get the embeddings of all images
    images = db.execute("SELECT paths,embedding FROM images").fetchall()
    distances = []
    for image in images:
        # convert the embedding from json to a tensor
        embedding = torch.tensor(json.loads(image[1])).to(device)
        # calculate the cosine distance between the prompt and the image
        distance = torch.cosine_similarity(text_features, embedding)
        # add the distance and the path to the list
        distances.append((distance, image[0]))
    # sort the list by distance
    distances.sort(key=lambda x: x[0], reverse=True)
    # print the top 10 results
    for i in range(10):
        print(distances[i][0], distances[i][1])

# read what does the user want to do
mode = sys.argv[1]
db = sqlite3.connect("images.db")

# if you have CUDA or MPS, set it to the active device like this
device = "cuda" if torch.cuda.is_available() else \
        ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"

# we initialize a tokenizer, image processor, and the model itself
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)

match mode:
    case "init":
        # we want to initialize the database
        db.execute("CREATE TABLE images (hash TEXT primary key, paths TEXT, embedding TEXT)")
        # create index for the hash column
        db.execute("CREATE INDEX hash_index ON images (hash)")
        db.execute("CREATE INDEX embedding_index ON images (embedding)")
        db.commit()
    case "import":
        # we want to read an image into the database
        import_image(sys.argv[2])
    case "import_dir":
        # we want to read all images in a directory into the database
        for currentpath, folders, files in os.walk(sys.argv[2]):
            for file in files:
                print(os.path.join(currentpath, file))
                import_image(os.path.join(currentpath, file))
    case "search":
        # we want to search for an image in the database
        search_image(sys.argv[2])