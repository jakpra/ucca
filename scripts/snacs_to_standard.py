import sys

from ucca import core as ucore, convert as uconv, layer0 as ul0, layer1 as ul1

from ucca.snacs import get_streusle_docs, get_passages, find_refined


def is_punct(token):
    return not any(c.isalnum() for c in token)

def create_terminal(tok, unit, l0, identified):
    term = l0.add_terminal(tok['word'], is_punct(tok['word']))
    for k, v in tok.items():
        if k == '#':
            term.extra['ind'] = v
        else:
            term.extra[k] = v
    term.extra['lexlemma'] = unit['lexlemma']
    term.extra['lexcat'] = unit['lexcat']
    term.extra.update(unit.get('heuristic_relation', {}))
    term.extra['is_part_of_mwe'] = len(unit['toknums']) > 1
    term.extra['identified_for_pss'] = int(identified)

    return term

def main(args):

    streusle_file = args[0]
    outpath = args[1]

    for doc_id, doc in get_streusle_docs(streusle_file).items():
        for unit in list(doc['exprs'].values()):
            ID = f'{doc_id}_{unit["sent_offs"]}_{unit["local_toknums"][0]}-{unit["local_toknums"][-1]}'
            sent = doc['sents'][int(unit['sent_offs'])-1]

            # print(sent)
            # print(unit)

            p = ucore.Passage(ID)
            l0 = ul0.Layer0(p)
            l1 = ul1.Layer1(p)

            root = l1.add_fnode(l1._head_fnode, ul1.EdgeTags.ParallelScene)

            # gov
            preterminal = l1.add_fnode(root, 'gov')
            # preterminal._fedge().attrib['remote'] = True
            if unit['heuristic_relation']['gov'] is not None:
                rel = sent['toks'][unit['heuristic_relation'][f'local_gov']-1]
                rel_unit = sent['swes'].get(str(rel['#']))
                if rel_unit is None:
                    rel_unit = sent['smwes'].get(str(rel.get('smwe', [-1, -1])[0]), None)
                term = create_terminal(rel, rel_unit, l0, False)
                preterminal.add(ul1.EdgeTags.Terminal, term)


            # P unit
            preterminal = l1.add_fnode(root, unit['ss'])
            for i in unit["toknums"]:
                tok = doc['toks'][i-1]
                term = create_terminal(tok, unit, l0, True)
                preterminal.add(ul1.EdgeTags.Terminal, term)

            # obj
            preterminal = l1.add_fnode(root, 'obj')
            # preterminal._fedge().attrib['remote'] = True
            if unit['heuristic_relation']['obj'] is not None and unit['lexcat'] != 'PP':
                rel = sent['toks'][unit['heuristic_relation'][f'local_obj'] - 1]
                rel_unit = sent['swes'].get(str(rel['#']))
                if rel_unit is None:
                    rel_unit = sent['smwes'].get(str(rel.get('smwe', [-1, -1])[0]), None)
                term = create_terminal(rel, rel_unit, l0, False)
                preterminal.add(ul1.EdgeTags.Terminal, term)


            uconv.passage2file(p, f'{outpath}/{ID}.xml')


if __name__ == '__main__':
    main(sys.argv[1:])

