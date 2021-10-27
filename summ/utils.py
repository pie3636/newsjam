from spacy.lang.fr.stop_words import STOP_WORDS

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


def get_keyword_sentences(doc):
    """
        Returns a list of keyword-only sentences for a given document
        Arguments:
            `doc`   The original document
        Returns:
            A list of the sentences in `doc`, converted into a list of keywords
    """
    sentences = []
    cur_sentence = []
    for sent in doc.sents:
        for token in sent:
            if not token.text.lower() in STOP_WORDS and not token.is_punct:
                cur_sentence.append(token.lemma_)
        sentences.append(cur_sentence)
        cur_sentence = []
    return sentences


def build_summary(top_scores, doc, sentences):
    """
        Builds a summary from the indices of the best sentences in the text for a given method
        Arguments:
            `top_scores`    An array containing the best sentence or word indices for each category (clusters, topics...)
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
    
    # Try to add each sentence to the summary, starting from the best one
    # and making sure to not go over a tweet's length
    sents_to_add = []
    summary_size = 0
    for i in top_sentences:
        full_sent = all_sents[i].text
        new_size = summary_size + len(full_sent)
        if summary_size + new_size <= 280:
            sents_to_add.append(i)
            summary_size += len(full_sent) + 1 # +1 because of the space/newline between sentences

    # Now that we have the optimal list of sentences,
    # build the actual summary as well as the keyword-only version
    summary = ''
    keyword_summary = ''
    for sent_idx in sents_to_add:
        keyword_sent = ' '.join(word for word in sentences[sent_idx])
        full_sent = all_sents[sent_idx].text
        keyword_summary += keyword_sent + '\n'
        summary += full_sent + '\n'

    # Remove the final space/newline
    if summary:
        summary = summary[:-1]
        keyword_summary = keyword_summary[:-1]
    return summary, keyword_summary