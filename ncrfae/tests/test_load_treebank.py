import unittest

from model import treebank_loader


class TestLoadPascalDataset(unittest.TestCase):
    def test_all_sentence_tokens_same_length_and_has_no_punctuation(self):
        l = ["arabic", "basque", "childes", "czech", "danish", "dutch", "english", "portuguese", "slovene", "swedish"]
        for c in l:
            train, dev, test = treebank_loader.load_pascal_dataset(c)
            sents = train + dev + test

            for sent in sents:
                words, cpos, fpos, upos, head = sent.words, sent.ctags, sent.ftags, sent.utags, sent.heads
                self.assertEqual(len(words), len(cpos))
                self.assertEqual(len(words), len(fpos))
                self.assertEqual(len(words), len(upos))
                self.assertEqual(len(words), len(head))

                # the only punctuation left is '$'s
                punc_index_list = filter(lambda x: upos[x] == ".", range(len(upos)))
                for i in punc_index_list:
                    self.assertEqual(words[i], '$')


if __name__ == '__main__':
    unittest.main()
