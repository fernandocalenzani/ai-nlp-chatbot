import re

from convokit import Corpus, download

import constants


def load_dataset():
    corpus = Corpus(filename=download("movie-corpus"))

    _utterances = corpus.utterances.values()
    utterances = {}
    conversations = []

    for utterance in _utterances:
        utterances[utterance.id] = utterance.text
        conv = utterance.get_conversation()
        conversations.append(conv._utterance_ids)

    corpus.print_summary_stats()

    return utterances, conversations


def get_questions_answers(utterances, conversation):
    quest = []
    ans = []

    for chat in conversation:
        for i in range(len(chat) - 1):
            ans.append(utterances[chat[i]])
            quest.append(utterances[chat[i + 1]])

    return quest, ans


def clean_text(text, abbreviations):
    text = text.lower()
    text = re.sub(r"[-()#/@;:<>=+?.|,]", "", text)

    for abbreviation, replacement in abbreviations.items():
        text = re.sub(re.escape(abbreviation), replacement,
                      text, flags=re.IGNORECASE)

    return text


def clean_data(data):
    data_clean = []
    for _data in data:
        data_clean.append(clean_text(
            _data, constants.abbreviations))

    return data_clean


def mapping_words(data):
    mapped = {}

    for _data in data:
        for _word in _data.split():
            if _word not in mapped:
                mapped[_word] = 1
            else:
                mapped[_word] += 1

    return mapped


def token_words(data, limit=20):
    id_words = {}
    n_word = 0

    for word, count in data.items():
        if count >= limit:
            id_words[word] = n_word
            n_word += 1
    print(id_words)
    return id_words


utterances, conversations = load_dataset()
_question, _answers = get_questions_answers(utterances, conversations)
questions = clean_data(_question)
answers = clean_data(_answers)
map_questions_words = mapping_words(questions)
map_ans_words = mapping_words(answers)

""" id_words_ques = token_words(map_questions_words)
id_words_ans = token_words(map_questions_words)
 """
