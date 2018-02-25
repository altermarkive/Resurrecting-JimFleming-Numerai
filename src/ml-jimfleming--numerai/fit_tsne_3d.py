import os

if __name__ == '__main__':
    perplexity = os.getenv('TSNE_PERPLEXITY', '30')
    perplexity = [int(value) for value in perplexity.split(',')]
    for i in range(len(perplexity)):
        os.system(
            'python3 /code/bh_tsne/prep_data.py {}'.format(perplexity[i]))
        os.system(
            '/code/bh_tsne/bh_tsne')
        os.system(
            'python3 /code/bh_tsne/prep_result.py {}'.format(perplexity[i]))
