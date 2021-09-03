# [pe]rturbable [t]ext [e]xplanations

Contains implementations of explanation methods LIME and IME and their modifications.
The modifications use a text generator to create more natural perturbations.
In total, there are two base methods (LIME, IME) and three modifications (LIME+LM, IME + internal LM, IME + external LM).

**NOTE**: breaking changes are very likely in the future (including a name change to match the repo), use with caution.

## Minimal example

```python
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator
from explain_nlp.methods.lime_lm import LIMEMaskedLMExplainer

# The explained model, in this case a fictional NLI classifier
model = InterpretableBertForSequenceClassification(
    model_name="my-finetuned-snli-model", tokenizer_name="my-finetuned-snli-model",
    batch_size=8, max_seq_len=41, device="cpu"
)

# The generator of the perturbations, in this case a pre-trained bert-base-uncased LM
generator = BertForMaskedLMGenerator(
    model_name="bert-base-uncased", tokenizer_name="bert-base-uncased",
    batch_size=8, max_seq_len=41, device="cpu",
    strategy="top_p", top_p=0.9
)

# shared_vocabulary=True skips the conversion between explained model, and generator representation
# Only use this if the vocabularies are shared, e.g., if the model and generator are both derived from "bert-base-uncased"
explainer = LIMEMaskedLMExplainer(model, generator=generator, kernel_width=1.0, 
                                  shared_vocabulary=True,
                                  return_samples=True, return_scores=True)


# Explain prediction for class 0 of interpreted model
res = explainer.explain_text(
    ("A shirtless man skateboards on a ledge.", "A man without a shirt."), label=0, 
    num_samples=100
)

# Explained (sub)words and their importance
print(res["input"])
print(res["importance"])

```

More advanced examples will be added in the future.