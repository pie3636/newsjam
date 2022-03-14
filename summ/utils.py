from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from .text_processing import Post

def get_top_sentences(top_scores, article_size):
    """
        Picks the ordered list of indices of the best sentences to summarize the text
        Arguments:
            `top_scores`   An array containing the top sentences (index and score) for each topic
                           Example: [[(3, 0.5), (4, 0.35), (1, 0.15)], [(6, 0.75), (1, 0.45), (2, 0.3)]]
            `article_size` The number of sentences in the original article
        Returns:
            A list of the indices of the top sentences
    """
    # Algorithm: First choose the best sentence of each topic
    # Then choose the second best sentence of each topic, then the third...
    # Keep going until the desired number of sentences has been reached
    top_sentences = []
    for i in range(article_size):
        for val in top_scores:
            if i >= len(val):
                continue
            elem = val[i][0]
            if isinstance(elem, (list, tuple)):
                elem = elem[0]
            if elem not in top_sentences:
                top_sentences.append(elem)
    return top_sentences


def get_keyword_sentences(doc, lang='fr'):
    """
        Returns a list of keyword-only sentences for a given document
        Arguments:
            `doc`   The original document
            `lang`  language of the document
                - default language is 'fr' (French) to instantiate French stop_words
                - other current option is 'en' (English) to instantiate English stop_words
        Returns:
            A list of the sentences in `doc`, converted into a list of keywords
    """
    if lang == 'fr':
        stop_words = fr_stop
    elif lang == 'en':
        stop_words = en_stop

    sentences = []
    cur_sentence = []
    for sent in doc.sents:
        for token in sent:
            if not token.text.lower() in stop_words and not token.is_punct:
                cur_sentence.append(token.lemma_)
        sentences.append(cur_sentence)
        cur_sentence = []
    return sentences


def build_summary(top_scores, doc, sentences, max_len=280):
    """
        Builds a summary from the indices of the best sentences in the text for a given method
        Arguments:
            `top_scores`    An array containing the best sentence indices for each category (clusters, topics...)
            `doc`           The original document to summarize
            `sentences`     A list of keyword-only sentences in the original document
        Returns a tuple containing:
            - The generated summary in text form
            - A keywords-only version of the generated summary
    """

    # Sort the tables so that they contain the words in decreasing score order
    for elem in top_scores:
        elem.sort(reverse=True, key=lambda x: x[1])

    # Get a list of all sentences in decreasing order of importance
    all_sents = list(doc.sents)
    top_sentences = get_top_sentences(top_scores, len(all_sents) + 1)

    # Generate optimal summary
    sents_to_add, summary_size = _build_summary(top_sentences, all_sents, max_len)
    summary, keyword_summary = _build_summary_2(sentences, all_sents, sents_to_add)
    
    # Trim summary
    post = Post()
    print('[Summary before post-processing]')
    print(summary)
    summary = post.rep_search(summary)
    print('[Keyword summary before post-processing]')
    print(keyword_summary)
    keyword_summary = post.rep_search(keyword_summary)
    print('[Done]')
    
    # Attempt adding new sentences
    if len(summary) < summary_size:
        sents_to_add, summary_size = _build_summary(top_sentences, all_sents, max_len, sents_to_add)
        summary, keyword_summary = _build_summary_2(sentences, all_sents, sents_to_add)
        print('[Summary before 2nd post-processing]')
        print(summary)
        summary = post.rep_search(summary)
        print('[Keyword summary before 2nd post-processing]')
        print(keyword_summary)
        keyword_summary = post.rep_search(keyword_summary)
        print('[Done]')
        
    # Remove the final space/newline
    if summary:
        summary = summary.strip()
        keyword_summary = keyword_summary.strip()
    return summary, keyword_summary


def _build_summary(top_sentences, all_sents, max_len, sents_to_add=None):
    """
        Auxillary function for summary building. Attempts to fit as many sentences
        into a summary as possible.
        Specifically, tries to add each sentence to the summary, starting
        from the best one         and making sure to not go over a tweet's length
        Arguments:
            `top_sentences` A list of sentence indices sorted in decreasing order of importance
            `all_sents`     All sentences from the original document
            `max_len`       The maximum length of the summary in characters
            `sents_to_add`  A list of sentence indices already added to the summary
        Returns a tuple containing:
            - The list of sentence indices to be contained in the summary
            - The length of the generated summary in characters
    """
    # Try to add each sentence to the summary, starting from the best one
    # and making sure to not go over a tweet's length
    if sents_to_add is None:
        sents_to_add = set()
        
    summary_size = 0
    for i in top_sentences:
        if i not in sents_to_add:
            full_sent = all_sents[i].text
            new_size = summary_size + len(full_sent)
            if summary_size + new_size <= max_len:
                sents_to_add.add(i)
                summary_size += len(full_sent) + 1 # +1 because of the space/newline between sentences
    return sents_to_add, summary_size

def _build_summary_2(sentences, all_sents, sents_to_add):
    """
        Auxillary function for summary building. Builds the full and keyword-only summary.
        Arguments:
            `sents_to_add`  A list of sentence indices to add to the summary
            `all_sents`     All sentences from the original document
            `sentences`     A list of keyword-only sentences in the original document
        Returns a tuple containing:
            - The generated summary in text form
            - A keywords-only version of the generated summary
    """
    summary = ''
    keyword_summary = ''
    for sent_idx in sorted(sents_to_add):
        keyword_sent = ' '.join(word for word in sentences[sent_idx])
        full_sent = all_sents[sent_idx].text
        keyword_summary += keyword_sent + '\n'
        summary += full_sent + '\n'
    return summary, keyword_summary