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
                  key=lambda e: e.child.position if type(e.child) == ul0.Terminal else e.child.start_position)

def find_in_siblings(terminal:ul0.Terminal, edge:ucore.Edge):
    try:
        return None if terminal is None else \
            next(sib for sib in siblings(edge) if terminal in sib.child.get_terminals())
    except StopIteration:
        return None

def get_text(node):
    return ' '.join([t.text for t in sorted(node.get_terminals(), key=lambda x: x.position)])


def heuristic_a(edge:ucore.Edge, *, gov_term=None, obj_term=None, **kwargs):
    '''
    Participant or circumstantial modifier of a scene
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param gov_term: terminal unit containing gov
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    parent = edge.parent
    if edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Function) and parent.fparent is not None and parent.fparent.is_scene:
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

def heuristic_b(edge:ucore.Edge, *, gov_term=None, obj_term=None, **kwargs):
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
            and (parent.fparent is None or not parent.fparent.is_scene) \
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

def heuristic_c(edge:ucore.Edge, *, lexcat='', obj_term=None, **kwargs):
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

def heuristic_d(edge:ucore.Edge, *, obj_term=None, **kwargs):
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

def heuristic_e(edge:ucore.Edge, *, lexcat='', **kwargs):
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

def heuristic_f(edge:ucore.Edge, *, lexcat='', **kwargs):
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

def heuristic_g(edge:ucore.Edge, *, lexcat='', **kwargs):
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

def heuristic_h(edge:ucore.Edge, ss, lexcat='', **kwargs):
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

def find_refined(term:ul0.Terminal, terminals:dict, local=False):

    if 'ss' not in term.extra or term.extra['ss'][0] != 'p':
        return [], {}

    successes_for_unit = fails_for_unit = 0
    failed_heuristics = []
    abgh = c = d = e = f = g = 0
    abgh_fail = c_fail = d_fail = e_fail = f_fail = g_fail = 0
    mwe_una_fail = no_match = 0
    warnings = 0

    ss = term.extra['ss']
    lexcat = term.extra['lexcat']
    # lexlemma = term.extra['lexlemma']
    toknums = sorted(map(int, str(term.extra[('local_' if local else '') + 'toknums']).split()))
    # span = f'{toknums[0]}-{toknums[-1]}'
    # rel = term.extra['heuristic_relation']
    gov, govlemma = term.extra.get(('local_' if local else '') + 'gov', -1), term.extra.get('govlemma', None)
    obj, objlemma = term.extra.get(('local_' if local else '') + 'obj', -1), term.extra.get('objlemma', None)
    pp_idiom = lexcat == 'PP'
    # if lexcat == 'PP':
    #     obj, objlemma = None, None
    config = term.extra['config']

    # terminals = dict(passage.layer('0').pairs)

    gov_term = terminals.get(gov, None)
    obj_term = terminals.get(obj, None)

    # try:
    #     unit_terminals = [terminals[toknum] for toknum in toknums]
    # except KeyError:
    #     print(toknums)
    #     print(terminals)
    #     exit(1)
    preterminals = term.parents
    if len(preterminals) != 1:
        # print(term.text, [str(pt) for pt in preterminals])
        return [], {}

    preterminal = preterminals[0]

    failed_heuristics = []

    # check whether SNACS mwe is UNA unit in UCCA
    # if len(toknums) > 1 and not pp_idiom:
    #     if not all(t.parents[0] == preterminal for t in unit_terminals[1:]):
    #         # skip SNACS unit if not all tokens are included in UCCA unit
    #         # fail(unit, None, f'terminals comprising strong MWE are not unanalyzable: [{lexlemma}] in {passage}')
    #         mwe_una_fail += 1
    #         fails_for_unit += 1
    #         failed_heuristics.append('MWE_UNA')
    #         return [], {} #'failed_heuristics':failed_heuristics}

    if len(preterminal.terminals) > len(toknums):
        # warn if UCCA UNA unit is larger than SNACS unit
        # warn(unit, None, f'PSS-bearing token(s) are part of a larger unanalyzable unit: [{lexlemma}] in {passage}')
        failed_heuristics.append('larger_UNA_warn')
        warnings += 1

    # assert len(preterminal.incoming) == 1, str(preterminal) + ' in ' + str(passage)
    refined = []

    for edge in preterminal.incoming:
        if edge is None:
            continue

        ref = None

        if edge.tag == ul1.EdgeTags.Center:
            if edge.parent._fedge() is not None:
               edge = edge.parent._fedge()

        # if edge.attrib.get('remote') or edge.child.attrib.get('implicit'):
        #     refined = None
        #     failed_heuristics.append('REM_IMP')

        if pp_idiom:
            ref = edge

        elif edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Connector, ul1.EdgeTags.Function):
            ref = heuristic_h(edge, ss, lexcat=lexcat) or \
                           heuristic_g(edge, lexcat=lexcat) or \
                           heuristic_a(edge, gov_term=gov_term, obj_term=obj_term) or \
                           heuristic_b(edge, gov_term=gov_term, obj_term=obj_term)

            abgh += 1
            if not ref:
                abgh_fail += 1
                failed_heuristics.append('ABGH')

        elif edge.tag in (ul1.EdgeTags.State, ul1.EdgeTags.Process):
            ref = heuristic_c(edge, lexcat=lexcat, obj_term=obj_term)

            c += 1
            if not ref:
                # print(term.extra)
                # input()
                c_fail += 1
                failed_heuristics.append('C')

        elif edge.tag == ul1.EdgeTags.Linker:
            ref = heuristic_d(edge, obj_term=obj_term)

            d += 1
            if not ref:
                d_fail += 1
                failed_heuristics.append('D')

        elif edge.tag in (ul1.EdgeTags.Adverbial, ul1.EdgeTags.Elaborator,
                          ul1.EdgeTags.Participant, ul1.EdgeTags.Time):
            ref = heuristic_e(edge, lexcat=lexcat)

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

        if ref is not None:
            refined.append(ref)

    error = {'abgh': abgh, 'c': c, 'd': d, 'e': e,
             'abgh_fail':abgh_fail, 'c_fail':c_fail, 'd_fail':d_fail, 'e_fail':e_fail,
             'no_match':no_match, 'mwe_una_fail':mwe_una_fail,
             'successes_for_unit':successes_for_unit, 'fails_for_unit':fails_for_unit,
             'warnings':warnings}
             # 'failed_heuristics':failed_heuristics}

    return (refined if len(refined) >= 1 else preterminal.incoming), error

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
                    expr['heuristic_relation']['local_gov'], expr['heuristic_relation']['gov'] = \
                        expr['heuristic_relation']['gov'], expr['heuristic_relation']['gov']+tok_offs
                if expr['heuristic_relation']['obj']:
                    expr['heuristic_relation']['local_obj'], expr['heuristic_relation']['obj'] = \
                        expr['heuristic_relation']['obj'], expr['heuristic_relation']['obj'] + tok_offs
                exprs[tuple(expr['toknums'])] = expr

        tok_offs += len(sent['toks'])

    return docs

def get_passages(streusle_file, ucca_path, annotate=True):

    v2_docids = set()
    with open(ucca_path + '/v2.txt') as f:
        for line in f:
            v2_docids.add(line.strip())

    for doc_id, doc in get_streusle_docs(streusle_file).items():
        ucca_file = ucca_path + '/xml/' + doc_id + '.xml'
        if doc_id not in v2_docids or not os.path.exists(ucca_file): continue

        passage = uconv.file2passage(ucca_file)

        tokens = [tok['word'] for tok in doc['toks']]
        terminals = passage.layer('0').pairs
        assert len(terminals) == len(
            tokens), f'unequal number of UCCA terminals and SNACS tokens: {terminals}, {tokens}'

        if annotate:
            for tok, (_, term) in zip(doc['toks'], terminals):
                for k, v in tok.items():
                    if k != '#':
                        term.extra[k] = v

            for unit in list(doc['exprs'].values()):
                terminal = terminals[unit['toknums'][0]-1][1]
                terminal.extra['ss'] = unit['ss']
                terminal.extra['ss2'] = unit['ss2']
                terminal.extra['toknums'] = ' '.join(map(str, unit['toknums']))
                terminal.extra['local_toknums'] = ' '.join(map(str, unit['local_toknums']))
                terminal.extra['lexlemma'] = unit['lexlemma']
                terminal.extra['lexcat'] = unit['lexcat']
                terminal.extra['config'] = unit['heuristic_relation']['config']
                terminal.extra.update(unit['heuristic_relation'])

        yield doc, passage
