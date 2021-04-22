import pandas as pd
import aspect_based_sentiment_analysis as absa
from tqdm import tqdm
import numpy as np

import html
from typing import List
from typing import Tuple
import numpy as np
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


def run_absa(subset, aspect):

    print('Running absa...')
    reviews = []

    for review in tqdm(subset['content']):
        if len(review) > 2078:
            continue

        run = nlp(text=review, aspects=[aspect])
        reviews.append(run.examples[0])

    return reviews


def retrieve_scores(absa_analysis):

    avg_distribution = np.sum(np.array(
        [example.scores for example in absa_analysis]), axis=0) / len(absa_analysis)
    neutral, negative, positive = avg_distribution

    return neutral, negative, positive


# df = pd.read_csv('combined_reviews_eng_CLEAN.csv')
df = pd.read_csv('combined_reviews_eng.csv')
# df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)

companies = df['company'].unique()
aspects = ['price', 'delivery', 'quality', 'return', 'refund',
           'product', 'service', 'customer service', 'order', 'experience']

recognizer = absa.aux_models.BasicPatternRecognizer()
nlp = absa.load(pattern_recognizer=recognizer)


def absa_iteration(company, aspect):

    print(f'Starting iteration for: {company} and {aspect} combination')
    some_dict = {}

    # Inputs
    some_dict['company'] = company
    some_dict['aspect'] = aspect

    # Subset
    subset = df[(df['company'] == company) & (
        df['content'].str.contains(aspect))]

    # Metrics
    some_dict['nb_reviews'] = len(subset)

    if len(subset) == 0:
        print(f"None of {company}'s reviews mention {aspect}...")
        print('Skip')
        some_dict['neutral'] = None
        some_dict['negative'] = None
        some_dict['positive'] = None
        some_dict['sentiment_scores'] = None

    else:
        # Run abstract based sentiment analysis
        absa_analysis = run_absa(subset, aspect)
        neutral, negative, positive = retrieve_scores(absa_analysis)

        some_dict['neutral'] = neutral
        some_dict['negative'] = negative
        some_dict['positive'] = positive

        # Plot input
        sentiment_scores = [review.scores[2] - review.scores[1]
                            for review in absa_analysis]
        some_dict['sentiment_scores'] = sentiment_scores

        # Examples

        str_to_remove = '<span style="background-color:rgba(180,180,180,1.1111111111111112);">Importance 1.00</span>'
        # Negative Examples

        idx_neg_5 = np.array([sent.scores[1]
                              for sent in absa_analysis]).argsort()[-10:][::-1]

        for idx, idx_neg in enumerate(idx_neg_5):

            neg_example = absa_analysis[idx_neg].scores

            print(f'iteration {idx} | score is {neg_example}')

            if neg_example[1] > 0.5:

                neg_html = display(absa_analysis[idx_neg].review).replace(
                    str_to_remove, '').replace('135,206,250', '242, 136, 136')

                soup = BeautifulSoup(neg_html.split('<br>')[
                                     0], features='lxml')

                length = len(
                    ' '.join([tag.string for tag in soup.find_all('span')]))

                text, margin = neg_html.split(
                    '<br>')[0], 25 + length // 100 * 25

                some_dict[f'neg_example{idx}'] = text
                some_dict[f'neg_example{idx}_margin'] = margin
            else:
                break

        # Positive Examples
        idx_pos_5 = np.array([sent.scores[2]
                              for sent in absa_analysis]).argsort()[-10:][::-1]

        for idx, idx_pos in enumerate(idx_pos_5):

            pos_example = absa_analysis[idx_pos].scores

            print(f'iteration {idx} | score is {pos_example}')

            if pos_example[2] > 0.5:

                pos_html = display(absa_analysis[idx_pos].review).replace(
                    str_to_remove, '').replace('135,206,250', '137, 242, 114')

                soup = BeautifulSoup(pos_html.split('<br>')[
                                     0], features='lxml')

                length = len(
                    ' '.join([tag.string for tag in soup.find_all('span')]))

                text, margin = pos_html.split(
                    '<br>')[0], 25 + length // 100 * 25

                some_dict[f'pos_example{idx}'] = text
                some_dict[f'pos_example{idx}_margin'] = margin
            else:
                break

    return some_dict


records = []

for company in companies:
    for aspect in aspects:
        records.append(absa_iteration(company, aspect))

print('Finished!')

new_df = pd.DataFrame(records)

# new_df.to_csv('TeStINg.csv')
new_df.to_csv('full-review-data.csv')
