import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from explain_nlp.experimental.core import MethodData
from explain_nlp.experimental.data import load_nli, LABEL_TO_IDX, TransformerSeqPairDataset
import pandas as pd

if __name__ == "__main__":
    MODEL_HANDLE = "/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased"
    TRAIN_DATA_PATH = "/home/matej/Documents/data/snli/snli_1.0_train.txt"
    DATA_PATH = "/home/matej/Documents/data/snli/snli_1.0_train_sample5000.txt"
    BATCH_SIZE = 8
    MAX_SEQ_LEN = 41
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MIN_SAMPLES_PER_FEATURE = 1

    tokenizer = BertTokenizer.from_pretrained(MODEL_HANDLE)
    model = BertForSequenceClassification.from_pretrained(MODEL_HANDLE, return_dict=True).to(DEVICE)
    model.eval()

    train_df = load_nli(TRAIN_DATA_PATH).sample(n=30)
    train_dataset = TransformerSeqPairDataset.build(train_df["sentence1"].values, train_df["sentence2"].values,
                                                    labels=train_df["gold_label"].apply(
                                                        lambda label_str: LABEL_TO_IDX["snli"][label_str]).values,
                                                    tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
    sample_data = train_dataset.input_ids

    test_df = load_nli(DATA_PATH).sample(n=10)
    test_dataset = TransformerSeqPairDataset.build(test_df["sentence1"].values, test_df["sentence2"].values,
                                                   labels=test_df["gold_label"].apply(
                                                       lambda label_str: LABEL_TO_IDX["snli"][label_str]).values,
                                                   tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)

    embeddings = []
    for i, curr_example in enumerate(tqdm(DataLoader(test_dataset, batch_size=1))):
        perturbable_mask = torch.logical_not(curr_example["special_tokens_mask"])[0]
        perturbable_inds = torch.arange(MAX_SEQ_LEN)[perturbable_mask]

        # Sample a permutation, current feature and random example to change the values to
        shuffled_inds = perturbable_inds[torch.randperm(perturbable_inds.shape[0])]
        random_feature = perturbable_inds[torch.randint(perturbable_inds.shape[0], ())]
        curr_feature_pos = int(torch.nonzero(shuffled_inds == random_feature, as_tuple=False).flatten())
        changed_indices = shuffled_inds[curr_feature_pos + 1:]

        idx_random_example = int(torch.randint(len(train_dataset), ()))

        pert_input_ids = curr_example["input_ids"]
        pert_input_ids[0, changed_indices] = sample_data[idx_random_example, changed_indices]

        with torch.no_grad():
            output = model.bert(input_ids=pert_input_ids.to(DEVICE),
                                **{k: v.to(DEVICE) for k, v in curr_example.items() if k in ["token_type_ids", "attention_mask"]})

        embeddings.append(output["pooler_output"].cpu())  # [curr_batch_size <= BATCH_SIZE, 768]

    embeddings = torch.cat(embeddings)
    embeddings = embeddings.numpy()

    cached_embeddings = pd.DataFrame(embeddings, columns=list(map(lambda i: f"a{i}",
                                                                  range(embeddings.shape[1]))))
    cached_embeddings.to_csv(f"snli_train5000_ime_bert_embeddings.csv", index=False)
