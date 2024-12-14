import json
import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import nltk
import random
import pickle
from collections import Counter
random.seed(42)


SEP = "<SEP>"


def _norm(x):
    return ' '.join(x.strip().split())


def process_data(i, d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = _norm(d['situation'])
    #init_intensity = int(d['score']['speaker']['begin_intensity'])
    #final_intensity = int(d['score']['speaker']['end_intensity'])

    dial = []
    for j, uttr in enumerate(d['dialog']):
        text = _norm(uttr['content'])
        role = uttr['speaker']
        if role == 'seeker':
            dial.append({
                'text': text,
                'speaker': 'usr',
            })
        else:
            dial.append({
                'text': text,
                'speaker': 'sys',
                'strategy': uttr['annotation']['strategy'],
            })

    res= {
        'dialog_id': i,
        'emotion_type': emotion,
        'problem_type': problem,
        'situation': situation,
        #'init_intensity': init_intensity,
        #'final_intensity': final_intensity,
        'dialog': dial
    }
    return res


strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
original = json.load(open('./ESConv.json'))

data = []
for i, d in enumerate(tqdm.tqdm(original, total=len(original))):
    data.append(process_data(i, d))

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion', emotions)
print('problem', problems)


random.seed(42)
random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
with open('./train.txt', 'w') as f:
    for e in train:
        f.write(json.dumps(e) + '\n')
with open('./sample.json', 'w') as f:
    json.dump(train[:10], f, ensure_ascii=False, indent=2)

print('valid', len(valid))
with open('./valid.txt', 'w') as f:
    for e in valid:
        f.write(json.dumps(e) + '\n')

print('test', len(test))
with open('./test.txt', 'w') as f:
    for e in test:
        f.write(json.dumps(e) + '\n')

print('all', len(data))
with open('./all.txt', 'w') as f:
    for e in data:
        f.write(json.dumps(e) + '\n')

# esconv_sentences
data_comet = {}
for dialog in data:
    context = [dialog['situation']]
    for u in dialog['dialog']:
        context.append(u['text'])
    data_comet[dialog['dialog_id']] = context
pickle.dump(data_comet, open('esconv_sentences.pkl', 'wb'))

# RECCON evaluation format
text, labels, id = [], [], []
for i, d in enumerate(original):
    prefix = [d['emotion_type'], SEP, _norm(d['situation']), SEP]
    dialog = [_norm(u["content"]) for u in d['dialog']]
    for j, u in enumerate(dialog):
        t = prefix.copy()
        t.extend([u, SEP])
        t.extend(dialog[:j+1] + [d['situation']])
        text.append(" ".join(t))
        labels.append(1)
        id.append(f"esconv.dialog_id_{i}_utt_{j+1}")

processed_data = {
    "text": text,
    "labels": labels,
    "id": id
}
df = pd.DataFrame(processed_data)
df.to_csv("esconv_classification_test_with_context.csv")