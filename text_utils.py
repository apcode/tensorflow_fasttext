from nltk.tokenize import word_tokenize


def TokenizeText(text):
    return word_tokenise(text.lower())


def ParseNgramsOpts(opts):
    ngrams = [int(g) for g in opts.split(',')]
    ngrams = [g for g in ngrams if (g > 1 and g < 7)]
    return ngrams


def GenerateNgrams(words, ngrams):
    nglist = []
    for ng in ngrams:
        for word in words:
            nglist.extend([word[n:n+ng] for n in range(len(word)-ng+1)])
    return nglist
