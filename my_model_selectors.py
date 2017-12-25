import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = None
        best_num_components = self.min_n_components
        best_bic = float('+inf')

        for num_states in range(self.min_n_components, self.max_n_components):

            try:
                # train model with training set
                hmm_model = GaussianHMM(n_components=num_states, n_iter=2000).fit(self.X, self.lengths)
                likelyhood = hmm_model.score(self.X, self.lengths)

                p = num_states ^ 2 + 2 * num_states * hmm_model.n_features - 1

                # now calculate bic
                bic = -2 * likelyhood + p * np.log(hmm_model.n_features)

                if bic < best_bic:
                    # new set of best numbers
                    best_num_components, best_bic, best_model = num_states, bic, hmm_model

            except Exception:
                # if it fails, it will try again with the next set of elements, or simply return an empty model
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = None
        best_num_components = self.min_n_components
        best_dic = float('-inf')

        # store it for later use
        logL_list = []

        for num_states in range(self.min_n_components, self.max_n_components):

            try:
                # train model with training set
                hmm_model = GaussianHMM(n_components=num_states, n_iter=2000).fit(self.X, self.lengths)
                likelyhood = hmm_model.score(self.X, self.lengths)

                logL_list.append((num_states, likelyhood, hmm_model))

            except Exception:
                # ignore the current exception, continue with next
                pass

        for i in range(0, len(logL_list) - 1):

            num_states, likelyhood, hmm_model = logL_list[i]

            sum_of_others = sum([likelyhood for idx, (num_states, likelyhood, hmm_model) in enumerate(logL_list) if idx != i])

            dic = likelyhood - 1 / (len(logL_list) - 1) * sum_of_others

            if dic > best_dic:
                best_num_components, best_dic, best_model = num_states, dic, hmm_model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        split_method = KFold()

        best_model = None
        best_num_components = self.min_n_components
        best_likelyhood = float('-inf')

        # somehow it works only for 2 or more
        if len(self.sequences) > 2:

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):


                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                for num_states in range(self.min_n_components, self.max_n_components):

                    try:
                        # train
                        hmm_model = GaussianHMM(n_components=num_states, n_iter=2000).fit(X_train, lengths_train)

                        likelyhood = hmm_model.score(X_test, lengths_test)

                        if likelyhood > best_likelyhood:

                            best_num_components, best_likelyhood, best_model = num_states, likelyhood, hmm_model

                    except Exception:
                        # if it fails, it will try again with the next set of elements, or simply return an empty model
                        pass

        else:

            # take first
            X_train, lengths_train = combine_sequences([0], self.sequences)

            if len(self.sequences) == 2:
                X_test, lengths_test = combine_sequences([1], self.sequences)
            else:
                X_test, lengths_test = combine_sequences([0], self.sequences)

            for num_states in range(self.min_n_components, self.max_n_components):

                try:
                    # train
                    hmm_model = GaussianHMM(n_components=num_states, n_iter=2000).fit(X_train, lengths_train)
                    likelyhood = hmm_model.score(X_test, lengths_test)

                    if likelyhood > best_likelyhood:
                        best_num_components, likelyhood, best_model = num_states, likelyhood, hmm_model

                except Exception:
                    # ignore current excetption
                    pass

        return best_model
