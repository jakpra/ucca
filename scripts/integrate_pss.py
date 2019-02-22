import sys, os
import json

import time

from collections import Counter

import matplotlib.pyplot as plt

from ucca import core as ucore, convert as uconv, layer0 as ul0, layer1 as ul1, visualization as uviz
from ucca.snacs import get_passages, find_refined

def main(args):
    try:
        integrate = True
        annotate = True
        object = False
        v2_only = True
        if '-I' in args:
            args.remove('-I')
            args.append('--no-integrate')
        if '--no-integrate' in args:
            integrate = False
            args.remove('--no-integrate')

        if '-A' in args:
            args.remove('-A')
            args.append('--no-annotate')
        if '--no-annotate' in args:
            integrate = False
            annotate = False
            args.remove('--no-annotate')

        if '-o' in args:
            args.remove('-o')
            args.append('--object')
        if '--object' in args:
            object = True
            args.remove('--object')

        if '--all' in args:
            v2_only = False
            args.remove('--all')

        streusle_file = args[0] #'../../streusle/streusle.govobj.json' #args[0] #'streusle.govobj.json'  # sys.argv[1]
        ucca_path = args[1] #'../../UCCA_English-EWT' #args[1] # '/home/jakob/nert/corpora/UCCA_English-EWT/xml'  # sys.argv[2]
        out_dir = args[2]

    except:
        print(f'usage: python3 {sys.argv[0]} STREUSLE_JSON UCCA_PATH OUT_DIR', file=sys.stderr)
        exit(1)

    with open(streusle_file) as f:
        streusle = json.load(f)

    print()

    unit_counter = 0
    successful_units = 0
    unsuccessful_units = 0

    doc_error = 0


    _doc_id = None

    v2_docids = set()
    if v2_only:
        with open(ucca_path + '/v2.txt') as f:
            for line in f:
                v2_docids.add(line.strip())

    ignore = """020851
                020992
                059005
                059416
                200957
                210066
                211797
                216456
                217359
                360937
                399348""".split()

    unit_times = []

    # print('usnacs.get_passages(streusle_file, ucca_path, annotate=(integrate or annotate), ignore=ignore, docids=v2_docids)')

    tag_refinements = Counter()

    for doc, passage, term2tok in get_passages(streusle_file, ucca_path, annotate=(integrate or annotate), target='obj' if object else 'prep', ignore=ignore, docids=v2_docids):
        if not integrate:
            for p in uconv.split_passage(passage, doc['ends'], map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
                uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')
            continue


        for pos, terminal in passage.layer('0').pairs:

            if 'ss' not in terminal.extra or terminal.extra['ss'][0] != 'p':
                # print(terminal.extra)
                continue

            # print('ok')

            start_time = time.time()
            unit_counter += 1

            refined, error = find_refined(terminal, dict(passage.layer(ul0.LAYER_ID).pairs))

            for r in refined:
                # TODO: deal with doubly refined edges
                if r.refinement:
                    pass
                else:
                    r.refinement = terminal.extra['ss']
                    tag_refinements[f'{r.tag}-{r.refinement}'] += 1

            # TODO
            if len(refined) >= 1:
                successful_units += 1

            # TODO
            else:
                unsuccessful_units += 1

                print('FAIL', doc['id'], terminal.extra['toknums'], terminal.extra['lexlemma'])


            unit_times.append(time.time() - start_time)


        for sent, psg in zip(doc['sents'], uconv.split_passage(passage, doc['ends'])):
            uviz.draw(psg)
            plt.savefig(f'../graphs/{sent["sent_id"]}.svg')
            plt.clf()

        for p in uconv.split_passage(passage, doc['ends'],
                                     map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
            uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')


    for x, y in tag_refinements.most_common(len(tag_refinements)):
        print(x, y, sep='\t')





    print(f'successful units\t{successful_units}\t{100*successful_units/(unit_counter-doc_error)}%')
    print(f'unsuccessful units\t{unsuccessful_units}\t{100-(100*successful_units/(unit_counter-doc_error))}%') #={unit_counter - doc_error - successful_units}={mwe_una_fail+abgh_fail+c_fail+d_fail+e_fail+f_fail+g_fail+no_match}



if __name__ == '__main__':
    #main(sys.argv[1:])
    #HOME = 'C:/Users/Jakob/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/jakob'
    #main([f'{HOME}/nert/streusle/streusle.govobj.json', f'{HOME}/nert/corpora/UCCA_English-EWT'])

    main(sys.argv[1:])
    # args = sys.argv[1:]
    #
    # try:
    #     integrate = True
    #     annotate = True
    #     if '-I' in args:
    #         args.remove('-I')
    #         args.append('--no-integrate')
    #     if '--no-integrate' in args:
    #         integrate = False
    #         args.remove('--no-integrate')
    #
    #     if '-A' in args:
    #         args.remove('-A')
    #         args.append('--no-annotate')
    #     if '--no-annotate' in args:
    #         integrate = False
    #         annotate = False
    #         args.remove('--no-annotate')
    #     streusle_file = args[0] #'../../streusle/streusle.govobj.json' #args[0] #'streusle.govobj.json'  # sys.argv[1]
    #     ucca_path = args[1] #'../../UCCA_English-EWT' #args[1] # '/home/jakob/nert/corpora/UCCA_English-EWT/xml'  # sys.argv[2]
    #     out_dir = args[2]
    #     #start_unit = int(args[-1])
    #     #streusle2ucca_file = '/home/jakob/nert/corpora/UCCA_English-EWT/streusle2ucca.txt'  # sys.argv[3]
    # except:
    #     print(f'usage: python3 {sys.argv[0]} STREUSLE_JSON UCCA_PATH OUT_DIR', file=sys.stderr)
    #     exit(1)
    #
    # with open(streusle_file) as f:
    #     streusle = json.load(f)
    #
    # print()
    #
    # unit_counter = 0
    # successful_units = 0
    # unsuccessful_units = 0
    # warnings = 0
    #
    # doc_error = 0
    #
    # fail_counts = Counter()
    #
    # n_figure = 0
    #
    # docs = {}
    # exprs = {}
    # toks = []
    # sents = []
    # ends = []
    # _doc_id = None
    # l_units = 0
    #
    # v2_docids = set()
    # # with open(ucca_path + '/v2.txt') as f:
    # #     for line in f:
    # #         v2_docids.add(line.strip())
    #
    # ignore = """020851
    #             020992
    #             059005
    #             059416
    #             200957
    #             210066
    #             211797
    #             216456
    #             217359
    #             360937
    #             399348""".split()
    #
    # unit_times = []
    #
    # print('usnacs.get_passages(streusle_file, ucca_path, annotate=(integrate or annotate), ignore=ignore, docids=v2_docids)')
    #
    # tag_refinements = Counter()
    #
    # for doc, passage, term2tok in usnacs.get_passages(streusle_file, ucca_path, annotate=True, target='prep'):  # , ignore=ignore, docids=v2_docids):
    #     if not integrate:
    #         for p in uconv.split_passage(passage, doc['ends'], map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
    #             uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')
    #         continue
    #
    #     # for _, unit in sorted(doc['exprs'].items()):
    #
    #     for pos, terminal in passage.layer('0').pairs:
    #
    #         if 'ss' not in terminal.extra or (terminal.extra['ss'][0] != 'p' and terminal.extra['ss'] != '`d'):
    #             # print(terminal.extra)
    #             continue
    #
    #         # print('ok')
    #
    #         start_time = time.time()
    #         unit_counter += 1
    #
    #         refined, error = usnacs.find_refined(terminal, dict(passage.layer(ul0.LAYER_ID).pairs))
    #
    #         # if doc_id == '231203' and unit['sent_offs'] == '0005':
    #         #     print(unit)
    #         #     print(passage)
    #         #     print(error)
    #
    #         for r in refined:
    #             # TODO: deal with doubly refined edges
    #             if r.refinement:
    #                 pass
    #             else:
    #                 r.refinement = terminal.extra['ss']
    #                 tag_refinements[f'{r.tag}-{r.refinement}'] += 1
    #
    #         # TODO
    #         if len(refined) >= 1:
    #             successful_units += 1
    #
    #             # uviz.draw(passage) #, highlit_ids=[n.ID for n in unit_terminals])
    #             # mng = plt.get_current_fig_manager()
    #             # mng.window.state("zoomed")
    #             # plt.show()
    #
    #             # print(passage)
    #             # if input('[N(ext)/q(uit)]').lower() in ('q', 'quit'):
    #             #     exit(0)
    #
    #         # TODO
    #         else:
    #             unsuccessful_units += 1
    #
    #             print('FAIL', doc['id'], terminal.extra['toknums'], terminal.extra['lexlemma'])
    #
    #             # print(unit)
    #             # print(error)
    #             # print(passage)
    #             # if input('[N(ext)/q(uit)]').lower() in ('q', 'quit'):
    #             #     exit(0)
    #
    #         #     for fail in failed_heuristics:
    #         #         fail_counts[fail] += 1
    #             # print(json.dumps(unit, indent=2))
    #             # # print(passage)
    #             # print(failed_heuristics)
    #             # n_figure += 1
    #             # # if n_figure >= 20:
    #             # #     plt.show()
    #             # #     n_figure = 1
    #             # # plt.figure(n_figure)
    #             # uviz.draw(passage, highlit_ids=[n.ID for n in unit_terminals])
    #             #
    #             # mng = plt.get_current_fig_manager()
    #             # mng.window.state("zoomed")
    #             # plt.show()
    #             # # node_link_data, pos = uviz.draw(passage), uviz.topological_layout(passage)
    #             # #with open('C:/Users/Jakob/Documents/nert/ucca-streusle/graphs/tmp.json', 'w') as f:
    #             # #    json.dump(node_link_data, f, indent=2)
    #             # #with open('C:/Users/Jakob/Documents/nert/ucca-streusle/graphs/tmp_pos.json', 'w') as f:
    #             # #    json.dump(pos, f, indent=2)
    #             # # show(node_link_data, pos)
    #             # # input('\nhit ENTER to continue...')
    #             # continue
    #
    #         # progress = 100 * unit_counter / l_units
    #         # print(f'{unit_counter} ({progress:.2f}%)')
    #
    #         unit_times.append(time.time() - start_time)
    #
    #
    #     for sent, psg in zip(doc['sents'], uconv.split_passage(passage, doc['ends'])):
    #         uviz.draw(psg)
    #         plt.savefig(f'../graphs/{sent["sent_id"]}.svg')
    #         plt.clf()
    #
    #     # with open(out_dir + '/' + ucca_file.rsplit('/', maxsplit=1)[1], 'w') as f:
    #
    #     for p in uconv.split_passage(passage, doc['ends'],
    #                                  map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
    #         uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')
    #
    #     # print(f'on avg {len(unit_times)/sum(unit_times)} u/s', file=sys.stderr)
    #
    #
    # for x, y in tag_refinements.most_common(len(tag_refinements)):
    #     print(x, y, sep='\t')
    #
    #
    #
    #
    #
    # # print('\n\n')
    # # print(f'total units\t{unit_counter}')
    # # # print(f'gov and obj present\t{gov_and_obj_counter}')
    # # print(f'document error\t{doc_error}\t{100*doc_error/unit_counter}%')
    # # print(f'document success\t{unit_counter - doc_error}\t{100-(100 * doc_error / unit_counter)}%')
    # # print('----------------------------------------------------')
    # print(f'successful units\t{successful_units}\t{100*successful_units/(unit_counter-doc_error)}%')
    # print(f'unsuccessful units\t{unsuccessful_units}\t{100-(100*successful_units/(unit_counter-doc_error))}%') #={unit_counter - doc_error - successful_units}={mwe_una_fail+abgh_fail+c_fail+d_fail+e_fail+f_fail+g_fail+no_match}
    # # print(f'warnings\t{global_error["warnings"]}')
    # # print('---------------------------------')
    # # for ftype, count in fail_counts.most_common():
    # #     print(f'{ftype}\t{count}')
    # # print('---------------------------------')
    # # print(f'\tMWE but not UNA\t{global_error["mwe_una_fail"]}')
    # # print(f'\tR ({global_error["abgh"]}) but A and B miss\t{global_error["abgh_fail"]}')
    # # print(f'\tS ({global_error["c"]}) but C miss\t{global_error["c_fail"]}')
    # # print(f'\tL ({global_error["d"]}) but D miss\t{global_error["d_fail"]}')
    # # print(f'\tD or E ({global_error["e"]}) but E miss\t{global_error["e_fail"]}')
    # # # print(f'\tA ({f}) but F miss\t{f_fail}')
    # # # print(f'\tF ({g}) but G miss\t{g_fail}')
    # # print(f'\tno match\t{global_error["no_match"]}') #\t{ucca_categories}')
    #
    #
    # # print('---------------------------------')
    # # print(f'\tdeductable (multiple fails or fail and success for single unit)\t{deductable_multiple_fails}')
    # # print(f'config considered {tuple(considered)}: {considered_counter}')
    # # print(f'linked as expected: {linked_as_expected_counter}')
    #
    #
    # # print(json.dumps(to_SNACS, indent=2))
    # # print(json.dumps(to_UCCA, indent=2))