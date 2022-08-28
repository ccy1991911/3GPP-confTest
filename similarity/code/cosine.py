from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

s1 = 'the encryption of NAS messages has been started between the MME and the UE'
s2 = 'the UE shall start the ciphering and deciphering of NAS messages'

sentences1 = [s1]
sentences2 = [s2]

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings1, embeddings2)

for i in range(len(sentences1)):
    print(cosine_scores[i][i])
