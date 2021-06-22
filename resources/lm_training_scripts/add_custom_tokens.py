"""
    Script to load and re-save a tokenizer after adding custom (special) tokens to it.
    Useful for setting up controlled language modeling (provide custom --tokenizer_name to run_mlm.py)
"""
from transformers import BertTokenizerFast

if __name__ == "__main__":
    # Example for SST2 controlled MLM
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({
        'additional_special_tokens': ["<NEGATIVE>", "<POSITIVE>"]
    })

    tokenizer.save_pretrained("bert-base-uncased-tokenizer-sst2")
