import os
import pickle
import pandas as pd
import numpy as np

from typing import Optional, Dict, Any, List

import spacy
from spacy import displacy
from spacy.tokens import Doc
from spacy.matcher import DependencyMatcher

import streamlit as st
from spacy_streamlit.util import load_model, process_text, get_svg, LOGO

# I should really split these functions into other files, but it is tricky
# because I haven't managed my vars well...


def my_parser(doc: Doc, sent_index) -> Dict[str, Any]:
    words = [
        {
            "text": w.text,
            "tag": w.pos_,
            "lemma": None,
        }
        for w in doc
    ]
    arcs = []
    for i, (coltype, parent, child) in extractions:
        if i == sent_index:
            if child.i < parent.i:
                arcs.append(
                    {"start": child.i, "end": parent.i, "label": coltype, "dir": "left"}
                )
            elif child.i > parent.i:
                arcs.append(
                    {"start": parent.i, "end": child.i, "label": coltype, "dir": "right"}
                )
    return {"words": words, "arcs": arcs}


def visualize_parser(docs: List[spacy.tokens.Doc], *, title: Optional[str] = None, key: Optional[str] = None) -> None:
    st.header(title)
    cols = st.beta_columns(2)
    num_parses = cols[1].select_slider('Number of Sentences to Visualise:', options=[0, 1, 2, 3, 4], value=1)
    vismode = cols[0].radio('Which Dependencies to Show', ('All', 'Collocation Candidates'))
    if num_parses >= 1:
        for num, sent in enumerate(docs):
            if num < num_parses:
                allparsed = displacy.parse_deps(sent)
                colparsed = my_parser(sent, num)
                html = displacy.render((allparsed if vismode == 'All' else colparsed), style="dep", manual=True)
                # Double newlines seem to mess with the rendering
                html = html.replace("\n\n", "\n")
                if len(docs) > 1:
                    st.markdown(f"> {sent.text}")
                st.write(get_svg(html), unsafe_allow_html=True)

# Cosmetics

st.sidebar.markdown(LOGO, unsafe_allow_html=True)
st.sidebar.title('Collocations')
st.sidebar.markdown('Collocations are arbitrarily conventionalized combinations of words. '
                    'This page demos a new method for automatically extracting collocation candidates using a dependency parser and a part-of-speech tagger. '
                    'Here, candidates are extracted from a user-provided sample text. '
                    'The same method has been applied at scale to extract collocation candidates from a reference corpus.'
                    'Statistical measures of association strength (AMs) were calculated for all the collocation candidates extracted from the reference corpus.'
                    'Although there is no clear cutoff, the most strongly associated candidates can be considered collocations.'
                    'This method may be used to assess a language learner\'s lexical proficiency.'
                    ' -- Langdon Holmes, California State University Long Beach')


# Select a Model, Load its Configuration

models = {"en_core_web_sm": 'Small', "en_core_web_lg": 'Large', "en_core_web_trf": 'Transformer-based'}
model_names = models
format_func = str
format_func = lambda name: models.get(name, name)
model_names = list(models.keys())
spacy_model = st.sidebar.selectbox(
    "Choose a Pre-trained NLP Model:",
    model_names,
    index=0,
    format_func=format_func,
)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()

st.sidebar.subheader("spaCy pipeline:")
desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code></p>"""
st.sidebar.markdown(desc, unsafe_allow_html=True)

# Initialize Matcher Generator

matcher = DependencyMatcher(nlp.vocab)
with open('matcherPatterns.pickle', 'rb') as fp:
    pattern_dict = pickle.load(fp)
for coltype, pattern in pattern_dict.items():
    matcher.add(coltype, pattern)

# Text Box
default_text = "You can enter some sentences here to see their dependency relationships. In the sidebar, you can choose which spaCy pipeline to use. Hit ctrl + enter to give it a whirl (and check out how each parser handles the first phrase of this sentence)."
st.title("Collocation Extractor")
text = st.text_area("Text to analyze", default_text, height=200)

# Process Text, then retokenize with collapsed punctuation, then split into sentence docs
doc = process_text(spacy_model, text)

def my_spans(doc):
    spans = []
    for word in doc[:-1]:
        if word.is_punct or not word.nbor(1).is_punct:
            continue
        start = word.i
        end = word.i + 1
        while end < len(doc) and doc[end].is_punct:
            end += 1
        span = doc[start:end]
        spans.append((span, word.tag_, word.lemma_, word.is_alpha, word.ent_type_))
    with doc.retokenize() as retokenizer:
        for span, tag, lemma, is_alpha, ent_type in spans:
            attrs = {"tag": tag, "lemma": lemma, "is_alpha": is_alpha, "ent_type": ent_type}
            retokenizer.merge(span, attrs=attrs)
    docs = [span.as_doc() for span in doc.sents]
    return docs


docs = my_spans(doc)

# Run Collocation Candidate Extractor on the Punctuation-Collapsed Spans
# Gets a Tuple( sentence index, ( coltype as string, parent as doc.token, child as doc.token))

extractions = []
for i, sent in enumerate(docs):
    for match_id, match in matcher(sent):
        parent = sent[match[0]]
        child = sent[match[1]]
        coltype = nlp.vocab.strings[match_id]
        extractions.append((i, (coltype, parent, child)))

# Get text version of candidates for dataframe manipulations

textextractions = []
for (sent, (coltype, parent, child)) in extractions:
    textextractions.append((coltype, parent.lemma_, parent.tag_, child.lemma_, child.tag_))

# Run the Visualizer

visualize_parser(docs, title='Visualize the Dependencies')

st.header("Collocation Candidate Statistics")

@st.cache
def my_calc(textextractions):
    if os.path.isfile('Bigram Stats.pickle'):
        df1 = pd.read_pickle('Bigram Stats.pickle')
    num_candidates = 7389634

    common_cols = ["Collocation Type", "Headword Lemma", "Headword Tag", "Dependent Word Lemma", "Dependent Word Tag"]

    df2 = pd.DataFrame(textextractions, columns=common_cols)

    df = pd.merge(df1, df2, on=common_cols, how='inner')
    df = df.assign(r_1=lambda x: x["o_11"] + x["o_21"],
                    c_1=lambda x: x['o_11'] + x['o_12'],
                    e_11=lambda x: x['r_1'] * x['c_1'] / num_candidates,
                    MI=lambda x: np.log2(x['o_11'] / x['e_11']))
    df = df.assign(T=lambda x: (x['o_11'] - x['e_11']) / np.sqrt(x['o_11']))

    output = df.loc[:, ("Headword Lemma", "Dependent Word Lemma", "T", "MI")]
    output['Reference Frequency'] = df['o_11']
    return(output)


output = my_calc(textextractions)
st.table(output)
