import json
with open("inv_wiki_tra.json","r",encoding='UTF-8') as f:
    data=json.load(f)
    print(data[0])

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
def QAmodel(question,paragraph):
    encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)

    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs)
    start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)

    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
        
        #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('/content/drive/MyDrive/BM25ODQA/indexes/wiki')
question="王建民事誰"
hits = searcher.search(question)
def search_paragragh(id):
    for i in data:
        if i["id"]==id:
            return i["contents"]
print(QAmodel(question,search_paragragh(hits[0].docid)))