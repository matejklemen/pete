from transformers import BertTokenizerFast

if __name__ == "__main__":
    with open("/home/matej/Documents/data/SST-2/lm_data/sst2_train_lm.txt", "r") as f:
        lines = list(map(lambda s: s.strip(), f.readlines()))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    encoded = tokenizer.batch_encode_plus(lines)

    lengths = [len(curr) for curr in encoded["input_ids"]]
    sorted_lengths = sorted(lengths)

    print(f"Min: {sorted_lengths[0]}")
    print(f"95th perc.: {sorted_lengths[int(0.95 * len(sorted_lengths))]}")
    print(f"99th perc.: {sorted_lengths[int(0.99 * len(sorted_lengths))]}")
    print(f"Max: {sorted_lengths[-1]}")
