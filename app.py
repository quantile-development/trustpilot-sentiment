import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from streamlit_metrics import metric_row

st.set_page_config(page_title='Quantile | Analysing Trustpilot Reviews',
                   page_icon='https://quantile.nl/favicon/favicon.ico')

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# HELPER FUNCTIONS
@st.cache
def reviews_categorical_df(sentiment_scores):

    scores = []

    for score in sentiment_scores:
        if score < -0.3:
            scores.append('negative')
        elif score > 0.3:
            scores.append('positive')
        else:
            scores.append('neutral')

    df = pd.DataFrame([{"count": scores.count('negative'), 'sentiment': 'negative'},
                       {"count": scores.count(
                           'neutral'), 'sentiment': 'neutral'},
                       {"count": scores.count('positive'), 'sentiment': 'positive'}])

    return df


@st.cache
def reviews_categorical_plot(df):

    fig = px.bar(df, x='sentiment', y='count', color='sentiment',
                 color_discrete_map={'negative': 'rgba(242, 136, 136,1.25)',
                                     'positive': 'rgba(137, 242, 114,1.25)',
                                     'neutral': 'rgb(131, 133, 132)'})

    return fig


# Title
st.title('Analysing Trustpilot Reviews')


@st.cache(allow_output_mutation=True)
def read_dataframe():
    return pd.read_csv('full-review-data.csv').drop('Unnamed: 0', axis=1)


data = read_dataframe()

# List of companies
companies = data['company'].unique()

selected_company = st.selectbox(
    label='Select company',
    options=companies,
    key='selection-1')

st.markdown("""
<style>
.custom-label {
    font-size: 0.8rem;
    color: rgb(38, 39, 48);
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="custom-label">Select an aspect you want to investigate</p>',
            unsafe_allow_html=True)


aspects = ['service', 'customer service', 'experience',
           'product', 'price', 'quality',
           'order', 'refund', 'return',
           'delivery']

col1 = st.beta_columns(3)
col2 = st.beta_columns(3)
col3 = st.beta_columns(3)
col4 = st.beta_columns(3)

buttons = []

for idx, aspect in enumerate(aspects):
    count = data[(data['company'] == selected_company) & (
        data['aspect'] == aspect)].reset_index(drop=True)['nb_reviews'][0]

    if idx < 3:
        buttons.append(col1[idx].button(f'{aspect} ({count})', key=idx))
    elif idx < 6:
        buttons.append(col2[idx % 3].button(f'{aspect} ({count})', key=idx))
    elif idx < 9:
        buttons.append(col3[idx % 3].button(f'{aspect} ({count})', key=idx))
    else:
        buttons.append(col4[idx % 3].button(f'{aspect} ({count})', key=idx))

# Select button pressed last
selected_aspect = None

for idx, button in enumerate(buttons):
    if button:
        selected_aspect = aspects[idx]
        break


# Filter dataframe
df = data[(data['company'] == selected_company)
          & (data['aspect'] == selected_aspect)].reset_index(drop=True)


if selected_aspect:
    # If aspect is not present for company
    if df['nb_reviews'][0] == 0:
        st.warning(
            f'Note: "{selected_aspect}" is not mentioned in any of the available reviews for {selected_company}.')
    else:

        sentiment_scores = [float(i)
                            for i in df['sentiment_scores'][0][1:-1].split(',')]

        scores = reviews_categorical_df(sentiment_scores)

        negative = scores[scores['sentiment'] == 'negative']['count'].values[0]
        neutral = scores[scores['sentiment'] == 'neutral']['count'].values[0]
        positive = scores[scores['sentiment'] == 'positive']['count'].values[0]

        st.header(f'Sentiment Distribution')
        st.write(f'#### For reviews containing the word: "{selected_aspect}"')
        metric_row(
            {
                "# Reviews": df['nb_reviews'][0],
                "Negative": np.round(negative, 3),
                "Neutral": np.round(neutral, 3),
                "Positive": np.round(positive, 3),
            }
        )

        fig = reviews_categorical_plot(scores)

        config = {'displayModeBar': False}

        st.plotly_chart(fig, use_container_width=False, config=config)

        st.header(f'Example reviews')

        neg_columns = df.filter(regex='neg_example').dropna(axis=1).columns
        with st.beta_expander(f'Negative reviews ({int(len(neg_columns)/2)})'):

            for idx in range(0, len(neg_columns), 2):
                components.html(df[neg_columns[idx]][0],
                                height=df[neg_columns[idx+1]][0])

        pos_columns = df.filter(regex='pos_example').dropna(axis=1).columns
        with st.beta_expander(f'Positive reviews ({int(len(pos_columns)/2)})'):

            for idx in range(0, len(pos_columns), 2):
                components.html(df[pos_columns[idx]][0],
                                height=df[pos_columns[idx+1]][0])
