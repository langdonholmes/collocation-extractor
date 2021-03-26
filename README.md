# colextractor

This is a simple hack of [spaCy](https://spacy.io) streamlit. It rewrites some of the dependency visualizer code to display a subset of the dependencies, rather than the full dependency tree. Also included is some code for calculating measures of association strength, see Evert (2016) for details.


# dependency labels

spaCy uses [ClearNLP Dependency Labels](https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md). matcherPatterns.pickle in this repository filters these dependencies into the POS pairing notation widely used in Applied Linguistics (NOUN + VERB, for example). 

# spacy-streamlit

This package contains utilities for visualizing [spaCy](https://spacy.io) models
and building interactive spaCy-powered apps with
[Streamlit](https://streamlit.io). It includes various building blocks you can
use in your own Streamlit app, like visualizers for **syntactic dependencies**,
**named entities**, **text classification**, **semantic similarity** via word
vectors, token attributes, and more.

