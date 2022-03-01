from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
import torch
from transformers import AutoModel, AutoTokenizer #, get_linear_schedule_with_warmup
from data import get_examples, TransformersData
from torch.utils.data import DataLoader
import numpy as np
import time
import random
from tqdm import tqdm
import json
import sys
import os
# import pandas as pd
    
# INPUTS
pretrained_transformers_model = sys.argv[1] # For example: "xlm-roberta-base"
seed = int(sys.argv[2])
max_seq_length = int(sys.argv[3]) # max length of a document (in tokens)
batch_size = int(sys.argv[4])
dev_ratio = float(sys.argv[5])

# repo path is needed to find the relevant files.
repo_path = os.path.abspath('./')
print('repo_path:', repo_path)
if repo_path.split('/')[-1] != 'transformers_text_classification':
    print('The absolute path to the repo is not correct. Double check! Exit ...')
    sys.exit()

# MUST SET THESE VALUES
train_filename = repo_path + "/data/odeuropa_benchmark_en_bio_shuffled_train_20220219.jsonl" # sys.argv[1]
test_filename = repo_path + "/data/odeuropa_benchmark_en_bio_shuffled_test_20220219.jsonl" # training time test
label_list = ["0", "1"]

mode_selection = 'predictfile'
if mode_selection == 'training': # proper training
    only_test, predict = False, False
elif mode_selection == 'testonly': # the test set will be processed and scores will be reported.
    only_test, predict = True, False
elif mode_selection == 'predictfile': # predict a new file and write predictions in the file
    only_test, predict = True, True
    test_filename = repo_path + "/data/predictions_bert-emotions.csv.jsonl"  # "/data/fairy_sentences.jsonl" # text to predict.
else:
    print('please set a mode: training, testonly, predictfile. Exit ...')
    sys.exit()
print('The mode is', mode_selection)
print('The file to be predicted is', test_filename)
    
has_token_type_ids = False # maybe old type BERT. calisiyorsa dokunma.

# SETTINGS
learning_rate = 2e-5
dev_metric = "f1_macro"
num_epochs = 30
dev_set_splitting = "random" # random, or any filename
use_gpu = True
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]


if use_gpu and torch.cuda.is_available():
    print('using GPU!')
    device = torch.device("cuda:%d"%(device_ids[0]))
else:
    print('using CPU!')
    device = torch.device("cpu")

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

tokenizer = None
model_path = "{}_{}_{}_{:.2f}_{}.pt".format(pretrained_transformers_model.replace("/", "_"), max_seq_length, batch_size, dev_ratio, seed)

perf_scores_path = model_path[:-3]+'.txt'

criterion = torch.nn.BCEWithLogitsLoss() if len(label_list) == 2 else torch.nn.CrossEntropyLoss(ignore_index=-1)

def test_model(encoder, classifier, dataloader):
    all_preds = []
    all_label_ids = []
    eval_loss = 0
    nb_eval_steps = 0
    for val_step, batch in enumerate(dataloader):
        if has_token_type_ids:
            input_ids, input_mask, token_type_ids, label_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
        else:
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask)[1]

        with torch.no_grad():
            out = classifier(embeddings)
            tmp_eval_loss = criterion(out.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()

        if len(label_list) == 2:
            curr_preds = torch.sigmoid(out).detach().cpu().numpy().flatten()
            curr_preds = [int(x >= 0.5) for x in curr_preds]
            all_preds += curr_preds
        else:
            out = out.detach().cpu().numpy()
            all_preds += np.argmax(out, axis=1).tolist()

        label_ids = label_ids.to('cpu').numpy().flatten().tolist()
        all_label_ids += label_ids

        nb_eval_steps += 1

    precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="macro", labels=list(range(0,len(label_list))))
    mcc = matthews_corrcoef(all_preds, all_label_ids)
    eval_loss /= nb_eval_steps
    result = {"precision_macro": precision,
              "recall_macro": recall,
              "f1_macro": f1,
              "mcc": mcc}

    return result, eval_loss


def model_predict(encoder, classifier, dataloader):
    all_preds = []
    for val_step, batch in enumerate(dataloader):
        if has_token_type_ids:
            input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
        else:
            input_ids, input_mask = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask)[1]

        with torch.no_grad(): # gerek olmayabilir, cunku eval mode enabled. ne olur ne olmaz.
            out = classifier(embeddings)

        if len(label_list) == 2:
            curr_preds = torch.sigmoid(out).detach().cpu().numpy().flatten()
            curr_preds = [int(x >= 0.5) for x in curr_preds]
            all_preds += curr_preds
        else:
            out = out.detach().cpu().numpy()
            all_preds += np.argmax(out, axis=1).tolist()

    return all_preds


def build_model(train_examples, dev_examples, pretrained_model, n_epochs=10, curr_model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    encoder = AutoModel.from_pretrained(pretrained_model)
    classifier = torch.nn.Linear(encoder.config.hidden_size, 1 if len(label_list) == 2 else len(label_list))

    train_dataset = TransformersData(train_examples, label_list, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_dataset = TransformersData(dev_examples, label_list, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size)


    classifier.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
    encoder.to(device)

    optimizer = torch.optim.AdamW(list(classifier.parameters()) + list(encoder.parameters()), lr=learning_rate)
    num_train_steps = int(len(train_examples) / batch_size * num_epochs)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps = 0,
    #                                             num_training_steps = num_train_steps)

    best_score = -1e6
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0
        encoder.train()
        classifier.train()

        print("Starting Epoch %d"%(epoch+1))
        global_step = 0
        train_loss = 0.0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if has_token_type_ids:
                input_ids, input_mask, token_type_ids, label_ids = tuple(t.to(device) for t in batch)
                embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
            else:
                input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
                embeddings = encoder(input_ids, attention_mask=input_mask)[1]

            out = classifier(embeddings)
            loss = criterion(out.view(-1), label_ids.view(-1))

            loss.backward()
            global_step += 1
            nb_tr_steps += 1

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0) # stabilizasyon. gradientlar kucuk norma.
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0) # stabilizasyon. gradientlar kucuk norma.
            optimizer.step()
            # scheduler.step()
            encoder.zero_grad()
            classifier.zero_grad()

            train_loss += loss.item()

        train_loss /= nb_tr_steps
        elapsed = time.time() - start_time

        # Validation
        encoder.eval()
        classifier.eval()
        result, val_loss = test_model(encoder, classifier, dev_dataloader)
        result["train_loss"] = train_loss
        result["dev_loss"] = val_loss
        result["elapsed"] = elapsed
        print("***** Epoch " + str(epoch + 1) + " *****")
        for key in sorted(result.keys()):
            print("  %s = %.6f" %(key, result[key]))

        print("Val score: %.6f" %result[dev_metric])

        if result[dev_metric] > best_score:
            print("Saving model!")
            torch.save(classifier.state_dict(), repo_path + "/models/classifier_" + curr_model_path)
            # encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # To handle multi gpu
            torch.save(encoder.state_dict(), repo_path + "/models/encoder_" + curr_model_path)
            best_score = result[dev_metric]

        print("========================================================================")

    return encoder, classifier

if __name__ == '__main__':
    if dev_set_splitting == "random":
        train_examples = get_examples(train_filename)
        random.shuffle(train_examples)
        dev_split = int(len(train_examples) * dev_ratio)
        dev_examples = train_examples[:dev_split]
        train_examples = train_examples[dev_split:]
    else: # it's a custom filename
        train_examples = get_examples(train_filename)
        random.shuffle(train_examples)
        dev_examples = get_examples(dev_set_splitting)

    if not only_test:
        encoder, classifier = build_model(train_examples, dev_examples, pretrained_transformers_model, n_epochs=num_epochs, curr_model_path=model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_transformers_model)
        encoder = AutoModel.from_pretrained(pretrained_transformers_model)
        classifier = torch.nn.Linear(encoder.config.hidden_size, 1 if len(label_list) == 2 else len(label_list))
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    classifier.load_state_dict(torch.load(repo_path + "/models/classifier_" + model_path))
    classifier.to(device)
    encoder.load_state_dict(torch.load(repo_path + "/models/encoder_" + model_path))
    encoder.to(device)
    encoder.eval()
    classifier.eval()

    if predict:
        test_examples = get_examples(test_filename, with_label=False)
        test_dataset = TransformersData(test_examples, label_list, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids, with_label=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        all_preds = model_predict(encoder, classifier, test_dataloader)
        with open(test_filename, "r", encoding="utf-8") as f:
            test = [json.loads(line) for line in f.read().splitlines()]

        with open(repo_path + "/out.json", "w", encoding="utf-8") as g:
            for i, t in enumerate(test):
                t["prediction"] = all_preds[i]
                g.write(json.dumps(t) + "\n")

    else:
        test_examples = get_examples(test_filename)
        test_dataset = TransformersData(test_examples, label_list, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        result, test_loss = test_model(encoder, classifier, test_dataloader)
        result["test_loss"] = test_loss

        print("***** TEST RESULTS *****")
        for key in sorted(result.keys()):
            print("  %s = %.6f" %(key, result[key]))

        print("TEST SCORE: %.6f" %result[dev_metric])
        
        with open(repo_path + "/performance_scores/perf_score_" + perf_scores_path, 'w') as fw:
            for key in sorted(result.keys()):
                fw.write("  %s = %.6f" %(key, result[key]))
                fw.write('\n')

            fw.write("TEST SCORE: %.6f" %result[dev_metric])
            fw.write('\n')
            