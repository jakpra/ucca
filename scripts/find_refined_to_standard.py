import sys

import operator

from ucca import core as ucore, convert as uconv, layer0 as ul0, layer1 as ul1, constructions as uconst
from ucca.layer1 import EdgeTags

from ucca.snacs import get_passages, find_refined

# from tupa.features.feature_extractor import head_terminal, height


EDGE_PRIORITY = {tag: i for i, tag in enumerate((
    EdgeTags.Center,
    EdgeTags.Connector,
    EdgeTags.ParallelScene,
    EdgeTags.Process,
    EdgeTags.State,
    EdgeTags.Participant,
    EdgeTags.Adverbial,
    EdgeTags.Time,
    EdgeTags.Elaborator,
    EdgeTags.Relator,
    EdgeTags.Function,
    EdgeTags.Linker,
    EdgeTags.LinkRelation,
    EdgeTags.LinkArgument,
    EdgeTags.Ground,
    EdgeTags.Terminal,
    EdgeTags.Punctuation,
))}


def is_punct(token):
    return not any(c.isalnum() for c in token)

def create_terminal(node, unit, l0, identified):
    hdt = head_terminal(node)
    if hdt is None: return
    term = l0.add_terminal(hdt.text, hdt.punct)
    # for k, v in tok.items():
    #     if k == '#':
    #         term.extra['ind'] = v
    #     else:
    #         term.extra[k] = v
    toks = [t.text for t in node.get_terminals()]
    term.extra['lexlemma'] = ' '.join(toks)
    term.extra['lexcat'] = node.ftag
    # term.extra.update(unit.get('heuristic_relation', {}))
    term.extra['is_part_of_mwe'] = len(toks) > 1
    term.extra['identified_for_pss'] = int(identified)

    return term

def head_terminal(node, *_):
    if isinstance(node, ul0.Terminal):
        return node
    return head_terminal_height(node)


def height(node, *_):
    return head_terminal_height(node, return_height=True)


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate


MAX_HEIGHT = 30


@static_vars(node=None, head_terminal=None, height=None)
def head_terminal_height(node, return_height=False):
    if node is not head_terminal_height.node:
        # print('ok')
        head_terminal_height.node = head_terminal_height.head_terminal = node
        head_terminal_height.height = 0
        while not isinstance(head_terminal_height.head_terminal, ul0.Terminal):  # Not a terminal
            # print(head_terminal_height.height)
            edges = [edge for edge in head_terminal_height.node.outgoing if not edge.attrib.get('remote') and not edge.child.attrib.get('implicit')]
            # print(edges)
            if not edges or head_terminal_height.height > MAX_HEIGHT:
                # print('should not happen', edges, head_terminal_height.height)
                head_terminal_height.head_terminal = head_terminal_height.height = None
                break
            head_terminal_height.node = head_terminal_height.head_terminal = min(edges, key=lambda edge: min(edge.tags, key=lambda t: EDGE_PRIORITY.get(t, 0))).child
            # print(head_terminal_height.head_terminal)
            head_terminal_height.height += 1
    return head_terminal_height.height if return_height else head_terminal_height.head_terminal




# def create_terminal(edge, terminal, l0):
#     # print(edge, edge.child)
#     hd = head_terminal(edge.child)
#     # print(hd)
#     if hd is None: print(edge, edge.child)
#     term = l0.add_terminal(hd.text, is_punct(hd.text))
#     term.extra['height'] = height(edge.child)
#     term.extra['lexcat'] = edge.tag
#     term.extra['lexlemma'] = ' '.join(sorted(edge.tags, key=lambda t: EDGE_PRIORITY.get(t, 0)))
#     term.extra['identified_for_pss'] = int(terminal in edge.child.get_terminals())
#    # print(term, term.extra)
#
#    return term

def get_all_descendants(node, remotes=False, visited=None):
    """Returns a list of all terminals under the span of this FoundationalNode.
    :param punct: whether to include punctuation Terminals, defaults to True
    :param remotes: whether to include Terminals from remote FoundationalNodes, defaults to false
    :param visited: used to detect cycles
    :return: a list of :class:`layer0`.Terminal objects
    """
    if not isinstance(node, ul1.FoundationalNode): # or isinstance(node, ul1.PunctNode):
        return []
    if visited is None:
        return sorted(get_all_descendants(node, remotes=remotes, visited=set()),
                      key=operator.attrgetter("start_position"))
    outgoing = {e for e in set(node) - visited if remotes or not e.attrib.get("remote")}
    return [n for e in outgoing for n in get_all_descendants(e.child, remotes=remotes, visited=visited | outgoing)] + [node]


def main(args):

    streusle_file = args[0]
    ucca_path = args[1]
    outpath = args[2]

    for doc, passage, term2tok in get_passages(streusle_file, ucca_path, annotate=True,
                                               target='prep'):


        sent_ids = map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])

        sent_passage = zip(sent_ids, uconv.split_passage(passage, doc['ends'], sent_ids))

        for sent, psg in sent_passage:

            p = uconv.join_passages([psg])
            l0 = p.layer(ul0.LAYER_ID)
            l1 = p.layer(ul1.LAYER_ID)

            for pos, terminal in l0.pairs:

                # print(terminal.extra)
                if 'ss' not in terminal.extra or not isinstance(terminal.extra['ss'], str) or terminal.extra['ss'][0] != 'p':
                    # print(terminal.extra)
                    continue

                unit = doc["exprs"][tuple(map(int, terminal.extra["toknums"].split()))]

                # pt = terminal.incoming[0].parent
                # node = pt.fparent
                # if node.fparent:
                #     node = node.fparent
                # nodes = set(get_all_descendants(node, remotes=True))

                # print(refined)

                # for n in nodes:
                ID = f'{doc["id"]}_{unit["sent_offs"]}_{unit["local_toknums"][0]}-{unit["local_toknums"][-1]}'

                # p = ucore.Passage(ID)
                # other_l0 = ul0.Layer0(p)
                # other_l1 = ul1.Layer1(p)
                #
                # root = other_l1.add_fnode(other_l1._head_fnode, ul1.EdgeTags.ParallelScene)
                #
                # # prep
                # term = create_terminal(pt, unit, other_l0, True)
                # if not term: continue
                # preterminal = other_l1.add_fnode(root, str(pt._fedge() in refined))
                # preterminal.add(ul1.EdgeTags.Terminal, term)
                #
                # # other node
                # term = create_terminal(n, unit, other_l0, False)
                # if not term: continue
                # preterminal = other_l1.add_fnode(root, str(n._fedge() in refined))
                # preterminal.add(ul1.EdgeTags.Terminal, term)


                refined, error = find_refined(terminal, dict(l0.pairs), local=True)

                for _, term in p.layer(ul0.LAYER_ID).pairs:
                    _pt = term.incoming[0].parent
                    toks = [t.text for t in _pt.get_terminals()]
                    term.extra['lexlemma'] = ' '.join(toks)
                    term.extra['lexcat'] = _pt.ftag
                    # term.extra.update(unit.get('heuristic_relation', {}))
                    term.extra['is_part_of_mwe'] = len(toks) > 1
                    term.extra['identified_for_pss'] = str(term.ID == terminal.ID)


                edges = [c.edge for cs in uconst.extract_candidates(p).values() for c in cs]
                for edge in edges:
                    edge.categories = []
                    edge.add(str(edge in refined))


                uconv.passage2file(p, f'{outpath}/{ID}.xml')



if __name__ == '__main__':
    main(sys.argv[1:])


