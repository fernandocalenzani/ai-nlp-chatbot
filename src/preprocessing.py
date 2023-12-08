import re

import numpy as np
from convokit import Corpus, download

import constants


class Preprocessing:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __load_dataset(self):
        corpus = Corpus(filename=download(self.dataset_name))

        _utterances = corpus.utterances.values()
        utterances = {}
        conversations = []

        for utterance in _utterances:
            utterances[utterance.id] = utterance.text
            conv = utterance.get_conversation()
            conversations.append(conv._utterance_ids)

        corpus.print_summary_stats()

        return utterances, conversations

    def __get_questions_answers(self, utterances, conversation):
        quest = []
        ans = []

        for chat in conversation:
            for i in range(len(chat) - 1):
                ans.append(utterances[chat[i]])
                quest.append(utterances[chat[i + 1]])

        return quest, ans

    def __clean_text(self, text, abbreviations):
        text = text.lower()
        text = re.sub(r"[-()#/@;:<>=+?.|,]", "", text)

        for abbreviation, replacement in abbreviations.items():
            text = re.sub(re.escape(abbreviation), replacement,
                          text, flags=re.IGNORECASE)

        return text

    def __clean_data(self, data):
        data_clean = []
        for _data in data:
            data_clean.append(self.__clean_text(
                _data, constants.abbreviations))

        return data_clean

    def __mapping_words(self, data):
        mapped = {}

        for _data in data:
            for _word in _data.split():
                if _word not in mapped:
                    mapped[_word] = 1
                else:
                    mapped[_word] += 1

        return mapped

    def __remove_words_by_low_frequency(self, data, limit=10):
        id_words = {}
        n_word = 0

        for word, count in data.items():
            if count >= limit:
                id_words[word] = n_word
                n_word += 1
        return id_words

    def __add_special_tokens_words(self, data, tokens):
        for token in tokens:
            data[token] = len(data) + 1
        return data

    def __invert_index_and_element(self, data):
        data = {p_i: p for p, p_i in data.items()}
        return data

    def __add_eos(self, data):
        for i in range(len(data)):
            data[i] += ' <EOS>'
        return data

    def __add_sos(self, data):
        for i in range(len(data)):
            data[i] = '<SOS> ' + data[i]
        return data

    def __encode_sentences(self, data, map_words):
        data_encoded = []

        for idata in data:
            ints = []
            for word in idata.split():
                if word not in map_words:
                    ints.append(map_words['<OUT>'])
                else:
                    ints.append(map_words[word])
            data_encoded.append(ints)

        return data_encoded

    def __order_list(self, data_qtn, data_ans, max_sentences):
        ordered_qtn = []
        ordered_ans = []

        for lenght in range(1, max_sentences):
            for i in enumerate(data_qtn):
                if len(i[1]) == lenght:
                    ordered_qtn.append(data_qtn[i[0]])
                    ordered_ans.append(data_ans[i[0]])

        return ordered_qtn, ordered_ans

    def add_padding(self, batch_txt, data):
        max_len = max([len(text) for text in batch_txt])

        return [text + [data['<PAD>']] * (max_len - len(text)) for text in batch_txt]

    def split_batches(self, qtn, ans, batch_size, qtn_data, ans_data):
        for batch_index in range(0, len(qtn) // batch_size):
            index_start = batch_index * batch_size

            qtn_in_batch = qtn[index_start:index_start + batch_size]
            ans_in_batch = ans[index_start:index_start + batch_size]

            qtn_in_batch_padd = np.array(
                self.add_padding(qtn_in_batch, qtn_data))
            ans_in_batch_padd = np.array(
                self.add_padding(qtn_in_batch, ans_data))
            yield qtn_in_batch_padd, ans_in_batch_padd



    def preprocessing(self):
        utterances, conversations = self.__load_dataset()
        _question, _answers = self.__get_questions_answers(
            utterances, conversations)

        questions = self.__clean_data(_question)
        answers = self.__clean_data(_answers)

        map_words_qtn = self.__mapping_words(questions)
        map_words_ans = self.__mapping_words(answers)

        """ map_words_qtn_filtered = self.remove_words_by_low_frequency(map_words_qtn)
        map_words_ans_filtered = self.remove_words_by_low_frequency(map_words_ans)
        """

        map_words_qtn = self.__add_special_tokens_words(
            map_words_qtn, constants.tokens)
        map_words_ans = self.__add_special_tokens_words(
            map_words_ans, constants.tokens)

        inverted_words_qtn = self.__invert_index_and_element(
            map_words_qtn)
        inverted_words_ans = self.__invert_index_and_element(
            map_words_ans)

        answers = self.__add_eos(answers)

        sentence_qtn_encoded = self.__encode_sentences(
            questions, map_words_qtn)
        sentence_ans_encoded = self.__encode_sentences(
            questions, map_words_ans)

        orderned_qtn, orderned_ans = self.__order_list(
            sentence_qtn_encoded, sentence_ans_encoded, 25)

        return orderned_qtn, orderned_ans
