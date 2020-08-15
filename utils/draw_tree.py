from typing import List, Dict, Tuple, Union, Any
from wsgiref import simple_server

from spacy.displacy import DependencyRenderer
# from spacy.util import prints

import dataloader
from treetk.dtree import DependencyTree
import spacy
from spacy import displacy


class DependencyTreeDrawer(object):
    def __init__(self, path: str):
        self.path = path
        self.rander = DependencyRenderer()
        self.html = ""

    @staticmethod
    def convert(edus: List[Tuple], arcs: List[Tuple]) -> Tuple[
        Dict[str, Union[List[Dict[str, str]], List[Dict[str, Union[str, Any]]]]], str]:

        format_arcs = []
        # arcs
        for head, dependent, label in arcs:
            if head > dependent:
                format_arcs.append({'start': dependent, 'end': head, 'label': label, 'dir': 'left'})
            else:
                format_arcs.append({'start': head, 'end': dependent, 'label': label, 'dir': 'right'})
        # words
        # word = [{"text": "EDU" + str(i), "tag": ""} for i, word in enumerate(tokens)]
        idx = [{"text": "EDU " + str(i), "tag": ""} for i, edu in enumerate(edus)]
        idx2edu = ["<li> EDU " + str(i) + ": " + " ".join(edu) + "</li>" for i, edu in enumerate(edus)]
        idx2edu_str = "<ul>\n" + "\n".join(idx2edu) + "\n</ul>"
        return {"words": idx, "arcs": format_arcs}, idx2edu_str

    def _html(self, parsed: List[Dict]):
        return self.rander.render(parsed).strip()

    # debug
    def show(self, edus: List[Tuple], arcs: List[Tuple]):
        parsed, edu_html = self.convert(edus, arcs)
        self.html = self._html([parsed])
        self.html += '\n' + edu_html
        httpd = simple_server.make_server('0.0.0.0', 5000, self.app)
        # prints("Using the '{}' visualizer".format('dep'),
        #        title="Serving on port {}...".format(5000))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Error")
            # prints("Shutting down server on port {}.".format(port))
        finally:
            httpd.server_close()

    def app(self, environ, start_response):
        # headers and status need to be bytes in Python 2, see #1227
        headers = [(str(b'Content-type', encoding='utf-8'),
                    str(b'text/html; charset=utf-8', encoding='utf-8'))]
        start_response(str(b'200 OK', encoding='utf-8'), headers)
        res = self.html.encode(encoding='utf-8')
        return [res]


# nlp = spacy.load("en_core_web_sm")
# text = """In ancient Rome, some neighbors live in three adjacent houses. In the center is the house of Senex, who lives there with wife Domina, son Hero, and several slaves, including head slave Hysterium and the musical's main character Pseudolus. A slave belonging to Hero, Pseudolus wishes to buy, win, or steal his freedom. One of the neighboring houses is owned by Marcus Lycus, who is a buyer and seller of beautiful women; the other belongs to the ancient Erronius, who is abroad searching for his long-lost children (stolen in infancy by pirates). One day, Senex and Domina go on a trip and leave Pseudolus in charge of Hero. Hero confides in Pseudolus that he is in love with the lovely Philia, one of the courtesans in the House of Lycus (albeit still a virgin)."""
# doc = nlp(text)
# sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style="dep", manual=True)

def main():
    # load scidtb
    train_dataset = dataloader.read_scidtb("train", "", relation_level="coarse-grained")
    drawer = DependencyTreeDrawer("")
    drawer.show(train_dataset[0].edus, train_dataset[0].arcs)

if __name__ == '__main__':
    main()