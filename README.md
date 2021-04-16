# colextractor

This is a simple hack of [spaCy](https://spacy.io) streamlit. It rewrites some of the dependency visualizer code to display a subset of the dependencies, rather than the full dependency tree. Also included is some code for calculating measures of association strength, see Evert (2016) for details.

## dependency labels

[spaCy](https://spacy.io) uses [ClearNLP Dependency Labels](https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md). `matcherPatterns.pickle` in this repository leverages spaCy's dependency matcher to filter the ClearNLP dependencies into the POS pairing notation widely used in Applied Linguistics (NOUN + VERB, for example). The schema is designed to align with the ACL (Ackermann and Chen, 2013), but also includes some additional combinations. The additional collocations can be easily filterd out later, or the schema can be customized.

| label        | headPOS | depPOS |
|--------------|---------|--------|
| NOUN + ADJ   | NOUN    | ADJ    |
| NOUN + NOUN  | NOUN    | NOUN   |
| VERB + NOUN  | VERB    | NOUN   |
| VERB + ADJ   | VERB    | ADJ    |
| VERB + ADV   | VERB    | ADV    |
| ADJ + ADV    | ADJ     | ADV    |
| NON-ACL below this point        |
| VERB + PART  | VERB    | PART   |
| VERB + PREP  | VERB    | ADP    |
| VERB + VERB  | VERB    | VERB   |
| NOUN + DET   | NOUN    | DET    |
| VERB + PREP  | VERB    | ADP    |
| PREP + NOUN  | ADP     | NOUN   |
| NOUN + PREP  | NOUN    | ADP    |
| VERB + AUX   | VERB    | AUX    |

Machine learning methods applied directly to spaCy's labelling schema could likely provide deeper insights here. Det + Noun combinations, for example, may be inversely correlated with writing quality, a feature that is overlooked with these schema.

## the study

To investigate word patterns and lexical development in the ICNALE corpus. These are argumentative essays written by English language learners.

## reference corpus

Currently, the software uses `OANCBigramStats.pickle`, a list of filtered dependency bigrams with co-occurrence frequencies pre-calculated. This bigram list is based on the Open American National Corpus. It seems to perform fairly well, but I will be making efforts to replace this corpus or add to it with more News and Gold-standard Student Papers -- published academic work and TV show subtitles are other corpus samples that might be helpful. However, the target corpus is exclusively argumentative essays, so I suspect a news corpus (particularly op-eds and opinion pieces) may be the most suitable for the task.

## docker

I also made a pullable docker image `docker pull langdonholmes/collocationextractor`. It is quite large (5.1GB), as it includes Python, PyTorch, several pre-trained NLP models, and all the relevant OANC dependency bigram statistics. I could get this down to probably 1-2 GB if I used only spaCy's 'small' pre-trained model, but the transformer-based model is significantly more accurate for distant dependencies, which are exactly the dependencies not captured by more traditional window/span collocation extraction methods, so there wouldn't be much point to all this with just the 'small' pre-trained model.

### from spacy-streamlit

This package contains utilities for visualizing [spaCy](https://spacy.io) models and building interactive spaCy-powered apps with [Streamlit](https://streamlit.io).

