import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    all_sequences = test_set.get_all_sequences()
    all_Xlenghts = test_set.get_all_Xlengths()

    print('Started recognizing ...')

    for i, test_word in zip(range(0, len(all_sequences)), test_set.wordlist):
        bestLL = float("-inf")
        bestWord = None
        probs = {}

        for word in models.keys():
            model = models[word]
            try:

                ll = model.score(all_sequences[i][0], all_Xlenghts[i][1])
                if ll > bestLL:
                    bestLL = ll
                    bestWord = word

            except Exception:
                #print("some exception occurred, ignoring")
                pass

            probs[word] = ll

        guesses.append(bestWord)
        probabilities.append(probs)

    print('Finished analyzing {} words '.format(len(all_sequences)))

    return probabilities, guesses
