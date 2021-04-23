
import streamlit as st


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


# Title
st.title('Analysing Trustpilot Reviews')


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
