import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_metrics import metric, metric_row
import aspect_based_sentiment_analysis as absa
from stqdm import stqdm
import numpy as np
import plotly.express as px

import html
from typing import List
from typing import Tuple
import numpy as np
from IPython.core.display import display as ipython_display
from IPython.core.display import HTML
from data_types import Pattern, PredictedExample, Review
from bs4 import BeautifulSoup


def html_escape(text):
    return html.escape(text)


def highlight(
        token: str,
        weight: float,
        rgb: Tuple[int, int, int] = (135, 206, 250),
        max_alpha: float = 0.8,
        escape: bool = True
) -> str:
    r, g, b = rgb
    color = f'rgba({r},{g},{b},{np.abs(weight) / max_alpha})'
    def span(c, t): return f'<span style="background-color:{c};">{t}</span>'
    token = html_escape(token) if escape else token
    html_token = span(color, token)
    return html_token


def highlight_sequence(
        tokens: List[str],
        weights: List[float],
        **kwargs
) -> List[str]:
    return [highlight(token, weight, **kwargs)
            for token, weight in zip(tokens, weights)]


def highlight_pattern(pattern: Pattern, rgb=(180, 180, 180)) -> str:
    w = pattern.importance
    html_importance = highlight(f'Importance {w:.2f}', w, rgb=rgb,
                                max_alpha=0.9)
    html_patterns = highlight_sequence(pattern.tokens, pattern.weights)
    highlighted_text = [html_importance, *html_patterns]
    highlighted_text = ' '.join(highlighted_text)
    return highlighted_text


def display_html(patterns: List[Pattern]):
    texts = []
    texts.extend([highlight_pattern(pattern) + '<br>' for pattern in patterns])
    text = ' '.join(texts)
    html_text = HTML(text)
    return text


def display_patterns(patterns: List[Pattern]):
    html_text = display_html(patterns)
    return html_text


def display(review: Review):
    return display_patterns(review.patterns)


def summary(example: PredictedExample):
    print(f'{str(example.sentiment)} for "{example.aspect}"')
    rounded_scores = np.round(example.scores, decimals=3)
    print(f'Scores (neutral/negative/positive): {rounded_scores}')

#############################################
#############################################
#############################################
#############################################
#############################################
#############################################

# FUNCTIONS
# @st.cache


def run_absa(aspect):

    sentences = []

    for sentence in stqdm(df['sentences']):

        run = nlp(text=sentence, aspects=[aspect])
        sentences.append(run.examples[0])

    return sentences


@st.cache
def retrieve_scores(absa_analysis):

    avg_distribution = np.sum(np.array(
        [example.scores for example in absa_analysis]), axis=0) / len(absa_analysis)

    neutral, negative, positive = avg_distribution

    return neutral, negative, positive


# LAYOUT / LOGIC
st.title('Analysing Trustpilot Reviews')

# Load the data
combined = pd.read_csv('combined_reviews_eng_CLEAN.csv')

# List of companies
companies = combined['company'].unique()

s1 = st.selectbox(
    label='Select company',
    options=companies,
    key='selection-1')


aspect = st.text_input(
    'Select an aspect you want to investigate')

df = combined[(combined['company'] == s1) & (
    combined['sentences'].str.contains(aspect))]

# submit = st.button('Run model')
submitted = False

if aspect:
    # # Reset submit button
    # submit = False

    if not s1:
        st.warning('Please select a company')

    elif not aspect:
        st.warning('Please select an aspect you want to analyze')

    elif len(df) == 0:
        st.warning(
            f'This aspect does not occur in the available reviews.')

    else:
        submitted = True

        # Aspect based sentiment analysis
        recognizer = absa.aux_models.BasicPatternRecognizer()
        nlp = absa.load(pattern_recognizer=recognizer)

        # Run abstract based sentiment analysis
        absa_analysis = run_absa(aspect)

if submitted:
    neutral, negative, positive = retrieve_scores(absa_analysis)

    st.header(f'Sentiment Distribution')
    st.write(f'#### For reviews containing the word: "{aspect}"')
    metric_row(
        {
            "# Reviews": len(df),
            "Negative": np.round(negative, 3),
            "Neutral": np.round(neutral, 3),
            "Positive": np.round(positive, 3),
        }
    )

    sentiment_scores = [review.scores[2] - review.scores[1]
                        for review in absa_analysis]

    fig = px.histogram(sentiment_scores, nbins=25)

    config = {'displayModeBar': False}

    fig.update_layout(
        xaxis_title_text='Sentiment',
        yaxis_title_text='Count',
        bargap=0.2,
        showlegend=False,
        dragmode=False,
        clickmode='none',
    )

    fig.update_traces(hovertemplate='<i><b>Count</i>: %{y}' +
                      '<br><b>Sentiment Range</b>: %{x}' +
                      '<br><extra></extra>')

    st.plotly_chart(fig, use_container_width=False, config=config)

    best_5 = np.array([sentence.scores[2]
                       for sentence in absa_analysis]).argsort()[-5:][::-1]

    worst_5 = np.array([sentence.scores[1]
                        for sentence in absa_analysis]).argsort()[-5:][::-1]

    st.header(f'Example Reviews')
    st.write(f'#### Negative example sentences')

    str_to_remove = '<span style="background-color:rgba(180,180,180,1.1111111111111112);">Importance 1.00</span>'

    for worst in worst_5:

        w_scores = absa_analysis[worst].scores

        if w_scores[1] > 0.5:

            worst_html = display(absa_analysis[worst].review).replace(
                str_to_remove, '').replace('135,206,250', '242, 136, 136')

            soup = BeautifulSoup(worst_html.split('<br>')[0])

            length = len(
                ' '.join([tag.string for tag in soup.find_all('span')]))

            components.html(worst_html.split('<br>')[
                            0], height=25+length//100 * 25)

        else:
            break

    st.write(f'#### Positive example sentences')

    for best in best_5:

        b_scores = absa_analysis[best].scores

        if b_scores[2] > 0.5:
            best_html = display(absa_analysis[best].review).replace(
                str_to_remove, '').replace('135,206,250', '137, 242, 114')

            soup = BeautifulSoup(best_html.split('<br>')[0])

            length = len(
                ' '.join([tag.string for tag in soup.find_all('span')]))

            components.html(best_html.split('<br>')[
                            0], height=25+length//100 * 25)
        else:
            break
