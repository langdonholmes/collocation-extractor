# colextractor

This is a simple hack of [spaCy](https://spacy.io) streamlit. It rewrites some of the dependency visualizer code to display a subset of the dependencies, rather than the full dependency tree. Also included is some code for calculating measures of association strength, see Evert (2016) for details.


# dependency labels

spaCy uses [ClearNLP Dependency Labels](https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md). matcherPatterns.pickle in this repository leverages spaCy's dependency matcher to filter the ClearNLP dependencies into the POS pairing notation widely used in Applied Linguistics (NOUN + VERB, for example). The schema is based on [Bhalla and Klimcikova (2019)](https://github.com/vishalbhalla/autocoleval), whose work is designed to align with the ACL (Ackermann and Chen, 2013).

# the study

To investigate word patterns and lexical development in the ICNALE corpus. These are argumentative essays written by English language learners.

# reference corpus

Currently, the software uses OANCBigramStats.pickle, a list of filtered dependency bigrams with co-occurrence frequencies pre-calculated. This bigram list is based on the Open American National Corpus. It seems to perform fairly well, but I will be making efforts to replace this corpus or add to it with more News and Gold-standard Student Papers -- published academic work and TV show subtitles are other corpus samples that might be helpful. However, the target corpus is exclusively argumentative essays, so I suspect a news corpus (particularly op-eds and opinion pieces) may be the most suitable for the task.

# spacy-streamlit

This package contains utilities for visualizing [spaCy](https://spacy.io) models
and building interactive spaCy-powered apps with
[Streamlit](https://streamlit.io). It includes various building blocks you can
use in your own Streamlit app, like visualizers for **syntactic dependencies**,
**named entities**, **text classification**, **semantic similarity** via word
vectors, token attributes, and more.

