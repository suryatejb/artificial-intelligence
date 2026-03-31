# This is for INFSCI 2440 in Spring 2026
# Please add comments with your code
# Task 1: Extract opinion pairs from reviews using Stanford CoreNLP (stanza)
# Dependency relations used:
#   nsubj: subject noun -> adjective head (e.g. "service is excellent" -> nsubj(excellent, service))
#   amod:  adjective modifier -> noun head (e.g. "warm atmosphere" -> amod(atmosphere, warm))
# Compound nouns are joined to form multi-word attributes (e.g. "waffle fries").

import stanza


class ExtractOpinions:
    # Extracted opinions and corresponding review id is saved in extracted_opinions, where KEY is the opinion
    # in "attribute, value" form (e.g. "service, good") and VALUE is the list of review_ids it appears in.

    def __init__(self):
        # Use instance-level dict so multiple calls accumulate correctly
        self.extracted_opinions = {}
        # Load stanza pipeline with tokenization, POS, lemma, and dependency parsing
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)

    def _get_full_noun(self, words, word_id, compounds):
        """Return the full noun phrase by prepending compound modifiers."""
        word = words[word_id - 1]  # word_id is 1-indexed
        prefix_parts = compounds.get(word_id, [])
        return ' '.join(prefix_parts + [word.text.lower()])

    def _add_opinion(self, review_id, attribute, value):
        """Add the (attribute, value) pair to extracted_opinions for this review."""
        opinion = attribute + ', ' + value
        if opinion not in self.extracted_opinions:
            self.extracted_opinions[opinion] = []
        if review_id not in self.extracted_opinions[opinion]:
            self.extracted_opinions[opinion].append(review_id)

    def extract_pairs(self, review_id, review_content):
        """Parse review_content and extract (attribute, value) opinion pairs."""
        doc = self.nlp(review_content)

        for sentence in doc.sentences:
            words = sentence.words

            # Build compound modifier map: head_id -> [modifier_text, ...]
            # Preserves left-to-right order so "waffle fries" stays "waffle fries"
            compounds = {}
            for word in words:
                if word.deprel == 'compound' and word.head > 0:
                    head_id = word.head
                    if head_id not in compounds:
                        compounds[head_id] = []
                    compounds[head_id].append(word.text.lower())

            # Build nsubj map: adj_word_id -> noun_word_id (for Rule 3 conj propagation)
            nsubj_map = {}
            for word in words:
                if word.head == 0:
                    continue
                head_word = words[word.head - 1]
                if (word.deprel == 'nsubj'
                        and word.upos in ('NOUN', 'PROPN')
                        and head_word.upos == 'ADJ'):
                    nsubj_map[word.head] = word.id  # adj_id -> subj_noun_id

            # Build amod map: adj_word_id -> noun_word_id (for Rule 4 conj propagation)
            # amod relation: adj -amod-> noun, so word is adj (dependent), word.head is noun
            amod_map = {}
            for word in words:
                if word.head == 0:
                    continue
                head_word = words[word.head - 1]
                if (word.deprel == 'amod'
                        and word.upos == 'ADJ'
                        and head_word.upos in ('NOUN', 'PROPN')):
                    amod_map[word.id] = word.head  # adj_id -> noun_id

            for word in words:
                if word.head == 0:
                    continue
                head_word = words[word.head - 1]

                # Rule 1 – nsubj: nominal subject is a noun, head is an adjective
                # Pattern: "service is excellent" -> service(NOUN) -nsubj-> excellent(ADJ)
                # Extracts: attribute = service, value = excellent
                # Use word.text (not lemma) for value to preserve word2vec vocabulary forms
                if (word.deprel == 'nsubj'
                        and word.upos in ('NOUN', 'PROPN')
                        and head_word.upos == 'ADJ'):
                    attribute = self._get_full_noun(words, word.id, compounds)
                    value = head_word.text.lower()
                    self._add_opinion(review_id, attribute, value)

                # Rule 2 – amod: adjectival modifier, head is a noun
                # Pattern: "warm atmosphere" -> warm(ADJ) -amod-> atmosphere(NOUN)
                # Extracts: attribute = atmosphere, value = warm
                elif (word.deprel == 'amod'
                      and word.upos == 'ADJ'
                      and head_word.upos in ('NOUN', 'PROPN')):
                    attribute = self._get_full_noun(words, word.head, compounds)
                    value = word.text.lower()
                    self._add_opinion(review_id, attribute, value)

                # Rule 3 – conj: adjective conjoined with another adjective that has a known nsubj
                # Pattern: "meat was tender and flavorful" -> flavorful -conj-> tender,
                #           tender has nsubj(meat), so extract [meat, flavorful]
                elif (word.deprel == 'conj'
                      and word.upos == 'ADJ'
                      and head_word.upos == 'ADJ'
                      and word.head in nsubj_map):
                    noun_id = nsubj_map[word.head]
                    attribute = self._get_full_noun(words, noun_id, compounds)
                    value = word.text.lower()
                    self._add_opinion(review_id, attribute, value)

                # Rule 4 – conj on amod: adjective conjoined with another adjective that
                # is an adjectival modifier (amod) of a noun.
                # Pattern: "fast and fresh food" -> fresh -conj-> fast, fast -amod-> food
                #           fast is in amod_map, so extract [food, fresh]
                elif (word.deprel == 'conj'
                      and word.upos == 'ADJ'
                      and head_word.upos == 'ADJ'
                      and word.head in amod_map):
                    noun_id = amod_map[word.head]
                    attribute = self._get_full_noun(words, noun_id, compounds)
                    value = word.text.lower()
                    self._add_opinion(review_id, attribute, value)
