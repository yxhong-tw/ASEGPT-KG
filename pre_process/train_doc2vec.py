import argparse
import gensim
import jieba
import os
import re

from datasets import load_dataset

if __name__ == "__main__":
    """
    If you want to train the Doc2Vec model with self-prepared data, the format of the data should be transform by `data_formatter.py`.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dn",
        "--data_number",
        default=0,
        type=int,
        help="The number of data to train the model (default: 0, all data)",
    )
    parser.add_argument(
        "-of",
        "--output_file",
        type=str,
        required=True,
        help="The output file path",
    )
    parser.add_argument(
        "-hdn",
        "--huggingface_dataset_name",
        type=str,
        default="zetavg/zh-tw-wikipedia",
        help="The name of huggingface dataset",
    )

    args = parser.parse_args()

    data = None

    dataset = load_dataset(args.huggingface_dataset_name)["train"]

    data = []

    for data_index, one_data in enumerate(dataset):
        if (data_index >= args.data_number and args.data_number != 0):
            break
        elif one_data["html"] == "" or one_data["html"] == None:
            continue

        data.append({
            "article_content": [
                re.sub(
                    r"<.*?>",
                    "",
                    one_data["html"],
                ),
            ]
        })

    print(f"Data {args.huggingface_dataset_name} loaded successfully!")

    model = gensim.models.Doc2Vec(
        vector_size=100,
        window=8,
        epochs=10,
        min_count=5,
        workers=os.cpu_count(),
    )

    tagged_document = gensim.models.doc2vec.TaggedDocument

    train_corpus = []

    for i in range(len(data)):
        x = data[i]["article_content"][0]

        x_tokenized = jieba.lcut(
            x,
            cut_all=False,
        )

        x_tagged = tagged_document(
            words=x_tokenized,
            tags=[i],
        )

        train_corpus.append(x_tagged)

    model.build_vocab(corpus_iterable=train_corpus)

    model.train(
        corpus_iterable=train_corpus,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    model.save(args.output_file)
