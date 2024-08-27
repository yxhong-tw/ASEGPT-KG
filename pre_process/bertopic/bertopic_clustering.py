import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from ckiptagger import WS, construct_dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForTokenClassification

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer


def load_embedding_model(
        model_name_or_path: str = '/root/ASEGPT-KG/shared_space/models/pre-process/model_ws'):
    return AutoModelForTokenClassification.from_pretrained(model_name_or_path)


def save_topic_figures(topic_model: BERTopic,
                       tot_data: pd.DataFrame,
                       fig_dir_path: str,
                       prefix_fig_name: str,
                       top_n_topics: int = 10):
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)

    # 各 Topic TF-IDF 關鍵字直方圖
    bar_fig = topic_model.visualize_barchart(
        top_n_topics=top_n_topics,
        width=200,
    )

    # 各 Topic 向量分佈圖
    topic_fig = topic_model.visualize_topics(top_n_topics=top_n_topics,
                                             width=1000,
                                             height=600)

    # 各 Topic 向量分佈圖
    tot_fig = topic_model.visualize_topics_over_time(tot_data,
                                                     top_n_topics=top_n_topics,
                                                     width=1000)

    bar_fig.write_html(
        f'{fig_dir_path}/{prefix_fig_name}_topic_{top_n_topics}-bar_fig.html')
    topic_fig.write_html(
        f'{fig_dir_path}/{prefix_fig_name}_topic_{top_n_topics}-topic_fig.html'
    )
    tot_fig.write_html(
        f'{fig_dir_path}/{prefix_fig_name}_topic_{top_n_topics}-tot_fig.html')


def main(articles_file_path: str,
         output_dir_path: str,
         docs_file_path: str,
         keywords_file_path: str,
         ckiptagger_model_path: str = './ckiptagger/data',
         use_labels: bool = False,
         num_topics: int = None):

    article_df = pd.read_json(articles_file_path)
    articles_content = article_df['article_content'].tolist()
    articles_timestamp = article_df['article_creation_date'].tolist()
    articles_category = article_df['source_category'].str.split(
        '>').str[-1].tolist()
    article_df['article_soruce_category'] = articles_category
    unique_categories = list(set(articles_category))
    category_mapping = {
        category: index
        for index, category in enumerate(unique_categories)
    }
    category_labels = [category_mapping[i] for i in articles_category]

    if os.path.exists(docs_file_path):
        print(f'Loading pre-tokenized documents from {docs_file_path}.')
        with open(docs_file_path, 'rb') as f:
            documents = pickle.load(f)
    else:
        with open(keywords_file_path) as file:
            keywords = file.read().splitlines()

        word_segmenter = WS(ckiptagger_model_path)
        word_sentence_list = word_segmenter(
            articles_content,
            sentence_segmentation=True,
            segment_delimiter_set={
                ',', '。', ':', '?', '!', ';', '、', '！', '？', '：', '，', '；', '‧'
            },
            # Words in this dictionary are encouraged
            recommend_dictionary=construct_dictionary({k: 1
                                                       for k in keywords}))

        # Convert to BERTopic acceptable format
        documents = [' '.join(w) for w in word_sentence_list]

        print(f'Saving tokenized documents to {docs_file_path}.')
        with open(docs_file_path, 'wb') as f:
            pickle.dump(documents, f)

    emb_model = load_embedding_model()
    vectorizer_model = CountVectorizer(stop_words='english')
    if use_labels:
        num_topics = len(unique_categories)

        empty_dimensionality_model = BaseDimensionalityReduction()
        clf = LogisticRegression()
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        topic_model = BERTopic(language='chinese (traditional)',
                               verbose=True,
                               embedding_model=emb_model,
                               vectorizer_model=vectorizer_model,
                               umap_model=empty_dimensionality_model,
                               hdbscan_model=clf,
                               ctfidf_model=ctfidf_model,
                               nr_topics=num_topics)
    else:
        topic_model = BERTopic(language='chinese (traditional)',
                               verbose=True,
                               embedding_model=emb_model,
                               vectorizer_model=vectorizer_model,
                               nr_topics=num_topics)

    pred_topics, pred_topic_probs = topic_model.fit_transform(
        documents, y=category_labels if use_labels else None)

    article_file_name = Path(articles_file_path).stem

    # Save the topic figures
    # save_topic_figures(topic_model=topic_model,
    #                    tot_data=topic_model.topics_over_time(
    #                        documents,
    #                        articles_timestamp,
    #                        pred_topics,
    #                        nr_bins=num_topics),
    #                    top_n_topics=num_topics,
    #                    fig_dir_path=f'{output_dir_path}/figs',
    #                    prefix_fig_name=article_file_name)

    num_topics = num_topics or 'auto'
    with open(
            f'{output_dir_path}/{article_file_name}-topic_info_{num_topics}.json',
            'w') as f:
        topic_info_df = topic_model.get_topic_info()

        if use_labels:
            # Assign original classes to our topics
            mappings = topic_model.topic_mapper_.get_mappings()
            mappings = {
                value: unique_categories[key]
                for key, value in mappings.items()
            }
            topic_info_df['Class'] = topic_info_df.Topic.map(mappings)

        topic_info_df.to_json(f, orient='records', force_ascii=False, indent=4)

    with open(f'{output_dir_path}/topic_{num_topics}_{article_file_name}.json',
              'w') as f:
        if use_labels:
            article_df['topic'] = article_df['topic'].map(mappings)
        else:
            topic_names = []
            for topic_id in pred_topics:
                info = topic_info_df[topic_info_df['Topic'] == topic_id].iloc[0]
                topic_names.append('_'.join(info['Representation']))

            article_df['topic'] = topic_names

        article_df['topic_id'] = pred_topics
        article_df.to_json(f, orient='records', force_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument(
        '-i',
        '--articles_file_path',
        type=str,
        required=True,
        help='Path to the articles file.')
    parser.add_argument('-o',
                        '--output_dir_path',
                        type=str,
                        required=True,
                        help='Path to the directory to save the output files.')

    # CKIPtagger
    parser.add_argument('-d',
                        '--docs_file_path',
                        type=str,
                        required=True,
                        help='Path to the documents file for saving the tokenized documents.')
    parser.add_argument('-k',
                        '--keywords_file_path',
                        type=str,
                        default='./pre_process/bertopic/data/keywords.txt',
                        help="Path to the keywords (don't split) file.")
    parser.add_argument('-m',
                        '--ckiptagger_model_path',
                        type=str,
                        default='./pre_process/bertopic/data/ckiptagger/data',
                        help='Path to the ckiptagger model.')

    # BERTopic
    try:
        parser.add_argument('--use_labels',
                            action=argparse.BooleanOptionalAction,
                            default=False,
                            help='Use labels for clustering.')
    except AttributeError:
        parser.add_argument('--use_labels',
                            action='store_true',
                            default=False,
                            help='Use labels for clustering.')
    parser.add_argument('--num_topics',
                        type=int,
                        default=None,
                        help='Number of topics to cluster. If no provided, will be automatically determined.')
    args = parser.parse_args()

    main(articles_file_path=args.articles_file_path,
         output_dir_path=args.output_dir_path,
         docs_file_path=args.docs_file_path,
         keywords_file_path=args.keywords_file_path,
         ckiptagger_model_path=args.ckiptagger_model_path,
         use_labels=args.use_labels,
         num_topics=args.num_topics)
