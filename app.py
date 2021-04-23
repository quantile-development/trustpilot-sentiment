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
# @st.cache
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


# @st.cache
def reviews_categorical_plot(df):

    fig = px.bar(df, x='sentiment', y='count', color='sentiment',
                 color_discrete_map={'negative': 'rgba(242, 136, 136,1.25)',
                                     'positive': 'rgba(137, 242, 114,1.25)',
                                     'neutral': 'rgb(131, 133, 132)'})

    return fig


# Title
st.title('Analysing Trustpilot Reviews')

st.markdown("""
### Created by [Quantile](https://quantile.nl)
###
""")


review_counts = {'booking.com': {'price': 14,
                                 'delivery': 0,
                                 'quality': 4,
                                 'return': 3,
                                 'refund': 29,
                                 'product': 0,
                                 'service': 34,
                                 'customer service': 23,
                                 'order': 1,
                                 'experience': 16},
                 'cheaptickets.nl': {'price': 6,
                                     'delivery': 1,
                                     'quality': 0,
                                     'return': 2,
                                     'refund': 24,
                                     'product': 0,
                                     'service': 31,
                                     'customer service': 12,
                                     'order': 2,
                                     'experience': 3},
                 'bol.com': {'price': 7,
                             'delivery': 26,
                             'quality': 3,
                             'return': 14,
                             'refund': 4,
                             'product': 23,
                             'service': 43,
                             'customer service': 21,
                             'order': 35,
                             'experience': 10},
                 'coolblue.nl': {'price': 7,
                                 'delivery': 35,
                                 'quality': 4,
                                 'return': 4,
                                 'refund': 2,
                                 'product': 19,
                                 'service': 33,
                                 'customer service': 9,
                                 'order': 16,
                                 'experience': 11},
                 'wehkamp.nl': {'price': 5,
                                'delivery': 26,
                                'quality': 3,
                                'return': 13,
                                'refund': 5,
                                'product': 10,
                                'service': 43,
                                'customer service': 27,
                                'order': 37,
                                'experience': 13},
                 'zalando.nl': {'price': 10,
                                'delivery': 14,
                                'quality': 5,
                                'return': 22,
                                'refund': 12,
                                'product': 7,
                                'service': 44,
                                'customer service': 25,
                                'order': 49,
                                'experience': 8},
                 'mediamarkt.nl': {'price': 8,
                                   'delivery': 19,
                                   'quality': 2,
                                   'return': 19,
                                   'refund': 16,
                                   'product': 20,
                                   'service': 50,
                                   'customer service': 28,
                                   'order': 50,
                                   'experience': 9},
                 'debijenkorf.nl': {'price': 7,
                                    'delivery': 13,
                                    'quality': 4,
                                    'return': 11,
                                    'refund': 10,
                                    'product': 7,
                                    'service': 34,
                                    'customer service': 21,
                                    'order': 31,
                                    'experience': 9}}


# List of companies
companies = ['booking.com', 'cheaptickets.nl', 'bol.com', 'coolblue.nl', 'wehkamp.nl',
             'zalando.nl', 'mediamarkt.nl', 'debijenkorf.nl']

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
    if idx < 3:
        buttons.append(col1[idx].button(
            f'{aspect} ({review_counts[selected_company][aspect]})', key=idx))
    elif idx < 6:
        buttons.append(col2[idx % 3].button(
            f'{aspect} ({review_counts[selected_company][aspect]})', key=idx))
    elif idx < 9:
        buttons.append(col3[idx % 3].button(
            f'{aspect} ({review_counts[selected_company][aspect]})', key=idx))
    else:
        buttons.append(col4[idx % 3].button(
            f'{aspect} ({review_counts[selected_company][aspect]})', key=idx))

# Select button pressed last
selected_aspect = None

for idx, button in enumerate(buttons):
    if button:
        selected_aspect = aspects[idx]
        break


# @st.cache(allow_output_mutation=True, max_entries=10, ttl=600)
def read_data(selected_company, selected_aspect):
    return pd.read_pickle('reviews-data.pkl').loc[selected_company, selected_aspect]


if selected_aspect:
    # Filter dataframe
    data = read_data(selected_company, selected_aspect)

    # If aspect is not present for company
    if data['nb_reviews'] == 0:
        st.warning(
            f'Note: "{selected_aspect}" is not mentioned in any of the available reviews for {selected_company}.')
    else:

        sentiment_scores = [float(i)
                            for i in data['sentiment_scores'][1:-1].split(',')]

        scores = reviews_categorical_df(sentiment_scores)

        negative = scores[scores['sentiment'] == 'negative']['count'].values[0]
        neutral = scores[scores['sentiment'] == 'neutral']['count'].values[0]
        positive = scores[scores['sentiment'] == 'positive']['count'].values[0]

        st.header(f'Sentiment Distribution')
        st.write(f'#### For reviews containing the word: "{selected_aspect}"')
        metric_row(
            {
                "# Reviews": data['nb_reviews'],
                "Negative": np.round(negative, 3),
                "Neutral": np.round(neutral, 3),
                "Positive": np.round(positive, 3),
            }
        )

        fig = reviews_categorical_plot(scores)

        config = {'displayModeBar': False}

        st.plotly_chart(fig, use_container_width=False, config=config)

        st.header(f'Inspect example reviews')

        neg_columns = data.filter(regex='neg_example').dropna().index
        with st.beta_expander(f'Negative reviews ({int(len(neg_columns)/2)})'):

            for idx in range(0, len(neg_columns), 2):
                components.html(data[neg_columns[idx]],
                                height=data[neg_columns[idx+1]])

        pos_columns = data.filter(regex='pos_example').dropna().index
        with st.beta_expander(f'Positive reviews ({int(len(pos_columns)/2)})'):

            for idx in range(0, len(pos_columns), 2):
                components.html(data[pos_columns[idx]],
                                height=data[pos_columns[idx+1]])
