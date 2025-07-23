from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

questions = list(faq_data.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)

def get_answer(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = model.similarity(question_embeddings, user_embedding)
    best_idx = similarities.argmax()
    return faq_data[questions[best_idx]]

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    answer = get_answer(user_input)
    print("Bot:", answer)