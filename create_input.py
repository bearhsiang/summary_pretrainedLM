import os
import json
import numpy as np

if __name__ == "__main__":
    data_dir = '/hdd/giga/'
    article_file = os.path.join(data_dir, 'train.article.txt')
    title_file = os.path.join(data_dir, 'train.title.txt')

    connect_token = ' <|sum|> '
    articles = []
    titles = []

    for article in open(article_file, 'r'):
        articles.append(article.strip())

    for title in open(title_file, 'r'):
        titles.append(title.strip())

    total = len(titles)

    output_dir = '/hdd/gpt2_sum/data'
    os.makedirs(output_dir, exist_ok=True)

    r = 0.7
    seed = np.arange(total)
    np.random.shuffle(seed)

    train_num = int(total*r)
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for i in seed[:train_num]:
            print(articles[i], titles[i], sep=connect_token, file=f)

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for i in seed[train_num:]:
            print(articles[i], titles[i], sep=connect_token, file=f)







        
