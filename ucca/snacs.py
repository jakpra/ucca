import sys, os
import json

import time

from collections import Counter

import matplotlib.pyplot as plt

from ucca import core as ucore, convert as uconv, layer0 as ul0, layer1 as ul1, visualization as uviz


def get_heads(node):
    if node.is_scene():
        return [node.process or node.state]
    else:
        return node.centers


def get_head_terminals(node):
    result = []
    agenda = [node]
    while agenda:
        current = agenda.pop()
        if current.terminals:
            result.extend(current.terminals)
        else:
            agenda.extend(get_heads(current))
    return sorted(result, key=lambda x: x.position)

def siblings(edge:ucore.Edge, exclude=False):
    return sorted([e for e in edge.parent.outgoing if not (exclude and edge.child.ID == e.child.ID)],
                  key=lambda e: e.child.start_position)

def find_in_siblings(terminal:ul0.Terminal, edge:ucore.Edge):
    try:
        return None if terminal is None else \
            next(sib for sib in siblings(edge) if terminal in sib.child.get_terminals())
    except StopIteration:
        return None

def get_text(node):
    return ' '.join([t.text for t in sorted(node.get_terminals(), key=lambda x: x.position)])


def heuristic_a(edge:ucore.Edge, passage:ucore.Passage, ss, gov_term=None, obj_term=None, **kwargs):
    '''
    Participant or circumstantial modifier of a scene
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param gov_term: terminal unit containing gov
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    parent = edge.parent
    if edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Function) and parent.fparent.is_scene:
        if parent.ftag in (ul1.EdgeTags.Process, ul1.EdgeTags.State):

            obj_edge = find_in_siblings(obj_term, parent._fedge())
            gov_edge = find_in_siblings(gov_term, parent._fedge())

            if gov_edge and gov_edge.tag == ul1.EdgeTags.Quantifier:
                return gov_edge
            else:
                return obj_edge

        else:
            sibs = siblings(edge)
            index_in_parent = sibs.index(edge)
            if (index_in_parent == 0 or index_in_parent == len(sibs) - 1):
                return parent._fedge()

def heuristic_b(edge:ucore.Edge, passage:ucore.Passage, ss, gov_term=None, obj_term=None, **kwargs):
    '''
    Configurative or circumstantial modifier of a non-scene
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param gov_term: terminal unit containing gov
    :param obj_term: terminal unit containing obj
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Connector, ul1.EdgeTags.Function):
        parent = edge.parent
        sibs = siblings(edge)
        index_in_parent = sibs.index(edge)
        if (index_in_parent == 0 or index_in_parent == len(sibs) - 1) \
            and not parent.fparent.is_scene \
            and parent.ftag != ul1.EdgeTags.Center:
            return parent._fedge()
        else:
            if parent.ftag == ul1.EdgeTags.Center:
                obj_edge = find_in_siblings(obj_term, parent._fedge())
                gov_edge = find_in_siblings(gov_term, parent._fedge())
            else:
                obj_edge = find_in_siblings(obj_term, edge)
                gov_edge = find_in_siblings(gov_term, edge)

            if gov_edge and gov_edge.tag == ul1.EdgeTags.Quantifier:
                return gov_edge
            else:
                return obj_edge

def heuristic_c(edge:ucore.Edge, passage:ucore.Passage, ss, lexcat='', obj_term=None, **kwargs):
    '''
    Predication
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :param obj_term: terminal unit containing obj
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag in (ul1.EdgeTags.State, ul1.EdgeTags.Process):
        if lexcat == 'PRON.POSS' or not obj_term:
            return edge
        else:
            return find_in_siblings(obj_term, edge)

def heuristic_d(edge:ucore.Edge, passage:ucore.Passage, ss, obj_term=None, **kwargs):
    '''
    Linkage
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param obj_term: terminal unit containing obj
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag == ul1.EdgeTags.Linker:
        return find_in_siblings(obj_term, edge)

def heuristic_e(edge:ucore.Edge, passage:ucore.Passage, ss, lexcat='', **kwargs):
    '''
    Intransitive prepositions, "particles", Possessive pronouns
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag in (ul1.EdgeTags.Adverbial, ul1.EdgeTags.Elaborator, ul1.EdgeTags.Participant, ul1.EdgeTags.Time):
        # if lexcat != 'PRON.POSS':
        return edge

def heuristic_f(edge:ucore.Edge, passage:ucore.Passage, ss, lexcat='', **kwargs):
    '''
    Participant possessive pronouns
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag == ul1.EdgeTags.Participant:
        if lexcat == 'PRON.POSS':
            return edge

def heuristic_g(edge:ucore.Edge, passage:ucore.Passage, ss, lexcat='', **kwargs):
    '''
    Adnominal infinitival purpose markers ("Inherent Purpose")
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag == ul1.EdgeTags.Function:
        if lexcat == 'INF.P':
            return edge.parent._fedge()

def heuristic_h(edge:ucore.Edge, passage:ucore.Passage, ss, lexcat='', **kwargs):
    '''
    Approximator
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if ss == 'p.Approximator':
        return edge

def find_refined(unit: dict, passage:ucore.Passage):

    if 'heuristic_relation' not in unit or unit['ss'][0] in '`?':
        raise ValueError

    successes_for_unit = fails_for_unit = 0
    failed_heuristics = []
    abgh = c = d = e = f = g = 0
    abgh_fail = c_fail = d_fail = e_fail = f_fail = g_fail = 0
    mwe_una_fail = no_match = 0
    warnings = 0

    ss = unit['ss']
    lexcat = unit['lexcat']
    lexlemma = unit['lexlemma']
    toknums = sorted(unit['toknums'])
    span = f'{toknums[0]}-{toknums[-1]}'
    rel = unit['heuristic_relation']
    gov, govlemma = rel.get('gov', -1), rel.get('govlemma', None)
    obj, objlemma = rel.get('obj', -1), rel.get('objlemma', None)
    pp_idiom = lexcat == 'PP'
    # if lexcat == 'PP':
    #     obj, objlemma = None, None
    config = rel['config']

    terminals = dict(passage.layer('0').pairs)

    gov_term = terminals.get(gov, None)
    obj_term = terminals.get(obj, None)

    try:
        unit_terminals = [terminals[toknum] for toknum in toknums]
    except KeyError:
        print(toknums)
        print(terminals)
        exit(1)
    preterminals = unit_terminals[0].parents
    assert len(preterminals) == 1

    preterminal = preterminals[0]

    failed_heuristics = []

    # check whether SNACS mwe is UNA unit in UCCA
    if len(toknums) > 1 and not pp_idiom:
        if not all(t.parents[0] == preterminal for t in unit_terminals[1:]):
            # skip SNACS unit if not all tokens are included in UCCA unit
            # fail(unit, None, f'terminals comprising strong MWE are not unanalyzable: [{lexlemma}] in {passage}')
            mwe_una_fail += 1
            fails_for_unit += 1
            failed_heuristics.append('MWE_UNA')
            return [], {} #'failed_heuristics':failed_heuristics}

    if len(preterminal.terminals) > len(toknums):
        # warn if UCCA UNA unit is larger than SNACS unit
        # warn(unit, None, f'PSS-bearing token(s) are part of a larger unanalyzable unit: [{lexlemma}] in {passage}')
        failed_heuristics.append('larger_UNA_warn')
        warnings += 1

    # assert len(preterminal.incoming) == 1, str(preterminal) + ' in ' + str(passage)
    refined = []

    for edge in preterminal.incoming:

        ref = None

        if edge.tag == ul1.EdgeTags.Center:
            edge = edge.parent._fedge()

        # if edge.attrib.get('remote') or edge.child.attrib.get('implicit'):
        #     refined = None
        #     failed_heuristics.append('REM_IMP')

        if pp_idiom:
            ref = edge

        elif edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Connector, ul1.EdgeTags.Function):
            ref = heuristic_h(edge, passage, ss, lexcat=lexcat) or \
                           heuristic_g(edge, passage, ss, lexcat=lexcat) or \
                           heuristic_a(edge, passage, ss, gov_term=gov_term, obj_term=obj_term) or \
                           heuristic_b(edge, passage, ss, gov_term=gov_term, obj_term=obj_term)

            abgh += 1
            if not ref:
                abgh_fail += 1
                failed_heuristics.append('ABGH')

        elif edge.tag in (ul1.EdgeTags.State, ul1.EdgeTags.Process):
            ref = heuristic_c(edge, passage, ss, lexcat=lexcat, obj_term=obj_term)

            c += 1
            if not ref:
                print(unit)
                input()
                c_fail += 1
                failed_heuristics.append('C')

        elif edge.tag == ul1.EdgeTags.Linker:
            ref = heuristic_d(edge, passage, ss, obj_term=obj_term)

            d += 1
            if not ref:
                d_fail += 1
                failed_heuristics.append('D')

        elif edge.tag in (ul1.EdgeTags.Adverbial, ul1.EdgeTags.Elaborator,
                          ul1.EdgeTags.Participant, ul1.EdgeTags.Time):
            ref = heuristic_e(edge, passage, ss, lexcat=lexcat)

            e += 1
            if not ref:
                e_fail += 1
                failed_heuristics.append('E')

        # elif edge.tag == ul1.EdgeTags.Participant:
        #     refined = heuristic_f(edge, passage, ss, lexcat=lexcat)
        #
        #     f += 1
        #     if not refined:
        #         f_fail += 1
        #         failed_heuristics.append('F')

        # elif edge.tag == ul1.EdgeTags.Function:
        #     refined = heuristic_g(edge, passage, ss, lexcat=lexcat)
        #
        #     g += 1
        #     if not refined:
        #         g_fail += 1
        #         failed_heuristics.append('G')

        else:
            no_match += 1
            failed_heuristics.append('__ALL__')

        if ref:
            refined.append(ref)

    error = {'abgh': abgh, 'c': c, 'd': d, 'e': e,
             'abgh_fail':abgh_fail, 'c_fail':c_fail, 'd_fail':d_fail, 'e_fail':e_fail,
             'no_match':no_match, 'mwe_una_fail':mwe_una_fail,
             'successes_for_unit':successes_for_unit, 'fails_for_unit':fails_for_unit,
             'warnings':warnings}
             # 'failed_heuristics':failed_heuristics}

    return refined, error

def get_streusle_docs(streusle_file):

    with open(streusle_file) as f:
        streusle = json.load(f)

    docs = {}
    exprs = {}
    _doc_id = None

    for sent in streusle:
        doc_id, sent_offs = sent['sent_id'].split('-')[-2:]

        if doc_id != _doc_id:
            tok_offs = 0
            if exprs:
                docs[_doc_id] = {'id': _doc_id, 'sents': sents, 'exprs': exprs, 'toks': toks, 'ends': ends}
            _doc_id = doc_id
            exprs = {}
            toks = []
            sents = []
            ends = []

        sents.append(sent)
        toks.extend(sent['toks'])
        ends.append(len(toks))

        for expr in list(sent['swes'].values()) + list(sent['smwes'].values()):
            if expr['ss'] and 'heuristic_relation' in expr:
                expr['sent_offs'] = sent_offs
                expr['doc_id'] = doc_id
                expr['local_toknums'] = expr['toknums']
                expr['toknums'] = [t + tok_offs for t in expr['toknums']]
                if expr['heuristic_relation']['gov']:
                    expr['heuristic_relation']['gov'] += tok_offs
                if expr['heuristic_relation']['obj']:
                    expr['heuristic_relation']['obj'] += tok_offs
                exprs[tuple(expr['toknums'])] = expr

        tok_offs += len(sent['toks'])

    return docs

def get_unit_passage(streusle_docs, ucca_path):

    v2_docids = set()
    with open(ucca_path + '/v2.txt') as f:
        for line in f:
            v2_docids.add(line.strip())

    for doc_id, doc in streusle_docs.items():
        ucca_file = ucca_path + '/xml/' + doc_id + '.xml'
        if doc_id not in v2_docids or not os.path.exists(ucca_file): continue

        passage = uconv.file2passage(ucca_file)

        tokens = [tok['word'] for tok in doc['toks']]
        terminals = dict(passage.layer('0').pairs)
        assert len(terminals) == len(
            tokens), f'unequal number of UCCA terminals and SNACS tokens: {terminals}, {tokens}'

        for unit in list(doc['exprs'].values()):
            yield (unit, passage)


# HOME = 'C:/Users/Jakob/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/jakob'
def main(args):
    try:
        streusle_file = args[0] #'../../streusle/streusle.govobj.json' #args[0] #'streusle.govobj.json'  # sys.argv[1]
        ucca_path = args[1] #'../../UCCA_English-EWT' #args[1] # '/home/jakob/nert/corpora/UCCA_English-EWT/xml'  # sys.argv[2]
        out_dir = args[2]
        #start_unit = int(args[-1])
        #streusle2ucca_file = '/home/jakob/nert/corpora/UCCA_English-EWT/streusle2ucca.txt'  # sys.argv[3]
    except:
        print(f'usage: python3 {sys.argv[0]} STREUSLE_JSON UCCA_PATH OUT_DIR', file=sys.stderr)
        exit(1)

    with open(streusle_file) as f:
        streusle = json.load(f)

    print()

    unit_counter = 0
    successful_units = 0
    unsuccessful_units = 0
    warnings = 0

    doc_error = 0

    fail_counts = Counter()

    n_figure = 0

    docs = {}
    exprs = {}
    toks = []
    sents = []
    ends = []
    _doc_id = None
    l_units = 0

    v2_docids = set()
    with open(ucca_path + '/v2.txt') as f:
        for line in f:
            v2_docids.add(line.strip())

    unit_times = []

    for doc_id, doc in get_streusle_docs(streusle_file).items():
        ucca_file = ucca_path + '/xml/' + doc_id + '.xml'
        if doc_id not in v2_docids or not os.path.exists(ucca_file): continue

        passage = uconv.file2passage(ucca_file)

        for _, unit in sorted(doc['exprs'].items()):

            if 'heuristic_relation' not in unit or unit['ss'][0] in '`?':
                continue

            start_time = time.time()
            unit_counter += 1

            refined, error = find_refined(unit, passage)

            if doc_id == '231203' and unit['sent_offs'] == '0005':
                print(unit)
                print(passage)
                print(error)

            for r in refined:
                if r.refinement:
                    pass
                else:
                    r.refinement = unit['ss']

            # TODO
            if len(refined) >= 1:
                successful_units += 1

                # uviz.draw(passage) #, highlit_ids=[n.ID for n in unit_terminals])
                # mng = plt.get_current_fig_manager()
                # mng.window.state("zoomed")
                # plt.show()

                # print(passage)
                # if input('[N(ext)/q(uit)]').lower() in ('q', 'quit'):
                #     exit(0)

            # TODO
            else:
                unsuccessful_units += 1

                print('FAIL', unit['doc_id'], unit['sent_offs'], unit['local_toknums'], unit['lexlemma'])

                # print(unit)
                # print(error)
                # print(passage)
                # if input('[N(ext)/q(uit)]').lower() in ('q', 'quit'):
                #     exit(0)

            #     for fail in failed_heuristics:
            #         fail_counts[fail] += 1
                # print(json.dumps(unit, indent=2))
                # # print(passage)
                # print(failed_heuristics)
                # n_figure += 1
                # # if n_figure >= 20:
                # #     plt.show()
                # #     n_figure = 1
                # # plt.figure(n_figure)
                # uviz.draw(passage, highlit_ids=[n.ID for n in unit_terminals])
                #
                # mng = plt.get_current_fig_manager()
                # mng.window.state("zoomed")
                # plt.show()
                # # node_link_data, pos = uviz.draw(passage), uviz.topological_layout(passage)
                # #with open('C:/Users/Jakob/Documents/nert/ucca-streusle/graphs/tmp.json', 'w') as f:
                # #    json.dump(node_link_data, f, indent=2)
                # #with open('C:/Users/Jakob/Documents/nert/ucca-streusle/graphs/tmp_pos.json', 'w') as f:
                # #    json.dump(pos, f, indent=2)
                # # show(node_link_data, pos)
                # # input('\nhit ENTER to continue...')
                # continue

            # progress = 100 * unit_counter / l_units
            # print(f'{unit_counter} ({progress:.2f}%)')

            unit_times.append(time.time() - start_time)


        for sent, psg in zip(doc['sents'], uconv.split_passage(passage, doc['ends'])):
            uviz.draw(psg)
            plt.savefig(f'..\graphs\{sent["sent_id"]}.svg')
            plt.clf()

        # with open(out_dir + '/' + ucca_file.rsplit('/', maxsplit=1)[1], 'w') as f:
        uconv.passage2file(passage, out_dir + '/' + ucca_file.rsplit('/', maxsplit=1)[1])

        # print(f'on avg {len(unit_times)/sum(unit_times)} u/s', file=sys.stderr)






    # print('\n\n')
    # print(f'total units\t{unit_counter}')
    # # print(f'gov and obj present\t{gov_and_obj_counter}')
    # print(f'document error\t{doc_error}\t{100*doc_error/unit_counter}%')
    # print(f'document success\t{unit_counter - doc_error}\t{100-(100 * doc_error / unit_counter)}%')
    # print('----------------------------------------------------')
    print(f'successful units\t{successful_units}\t{100*successful_units/(unit_counter-doc_error)}%')
    print(f'unsuccessful units\t{unsuccessful_units}\t{100-(100*successful_units/(unit_counter-doc_error))}%') #={unit_counter - doc_error - successful_units}={mwe_una_fail+abgh_fail+c_fail+d_fail+e_fail+f_fail+g_fail+no_match}
    # print(f'warnings\t{global_error["warnings"]}')
    # print('---------------------------------')
    # for ftype, count in fail_counts.most_common():
    #     print(f'{ftype}\t{count}')
    # print('---------------------------------')
    # print(f'\tMWE but not UNA\t{global_error["mwe_una_fail"]}')
    # print(f'\tR ({global_error["abgh"]}) but A and B miss\t{global_error["abgh_fail"]}')
    # print(f'\tS ({global_error["c"]}) but C miss\t{global_error["c_fail"]}')
    # print(f'\tL ({global_error["d"]}) but D miss\t{global_error["d_fail"]}')
    # print(f'\tD or E ({global_error["e"]}) but E miss\t{global_error["e_fail"]}')
    # # print(f'\tA ({f}) but F miss\t{f_fail}')
    # # print(f'\tF ({g}) but G miss\t{g_fail}')
    # print(f'\tno match\t{global_error["no_match"]}') #\t{ucca_categories}')


    # print('---------------------------------')
    # print(f'\tdeductable (multiple fails or fail and success for single unit)\t{deductable_multiple_fails}')
    # print(f'config considered {tuple(considered)}: {considered_counter}')
    # print(f'linked as expected: {linked_as_expected_counter}')


    # print(json.dumps(to_SNACS, indent=2))
    # print(json.dumps(to_UCCA, indent=2))


if __name__ == '__main__':
    #main(sys.argv[1:])
    #HOME = 'C:/Users/Jakob/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/jakob'
    #main([f'{HOME}/nert/streusle/streusle.govobj.json', f'{HOME}/nert/corpora/UCCA_English-EWT'])
    main(sys.argv[1:])