import wbia  # NOQA
# from wbia.other.detectfuncs import general_get_imageset_gids
from os.path import join, abspath, exists
import matplotlib.pyplot as plt
from wbia import plottool as pt
import numpy as np
import utool as ut
import tqdm
# import numpy as np
import datetime
import random

# ibs = None

ALL_VS_ALL = True
MAX_RANK = 20
MIN_AIDS_PER_NAME = 2
MAX_AIDS_PER_NAME = np.inf
MIN_TIME_DELTA = None
SEED = 21
# Checking 4165 annotations for 2601 names
# Using    2524 annotations for 960 names

# ALL_VS_ALL = True
# MAX_RANK = 12
# MIN_AIDS_PER_NAME = 2
# MAX_AIDS_PER_NAME = np.inf
# MIN_TIME_DELTA = None
# SEED = 2
# # Checking 4165 annotations for 2601 names
# # Using    2524 annotations for 960 names


def encounter_deltas(unixtimes):
    assert None not in unixtimes
    # assert -1 not in unixtimes
    previous = -1 * np.inf
    delta_list = []
    for unixtime in unixtimes + [None]:
        if unixtime is None:
            break
        else:
            try:
                delta = unixtime - previous
            except:
                delta = 0
        # print(delta)
        assert delta >= 0
        delta_list.append(delta)
        previous = unixtime
    assert len(delta_list) == len(unixtimes)
    delta_list = np.array(delta_list)
    return delta_list


def rank(ibs, result, cm_key=None):
    cm_dict = result['cm_dict']
    if cm_key is None:
        cm_key = list(cm_dict.keys())
        assert len(cm_key) == 1
        cm_key = cm_key[0]
    cm = cm_dict[cm_key]

    query_name = cm['qname']
    qnid = ibs.get_name_rowids_from_text(query_name)

    annot_uuid_list = cm['dannot_uuid_list']
    annot_score_list = cm['annot_score_list']
    daid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    dnid_list = ibs.get_annot_nids(daid_list)
    dscore_list = sorted(zip(annot_score_list, dnid_list), reverse=True)

    annot_ranks = []
    for rank, (dscore, dnid) in enumerate(dscore_list):
        if dnid == qnid:
            annot_ranks.append(rank)

    name_list = cm['unique_name_list']
    name_score_list = cm['name_score_list']
    dnid_list = ibs.get_name_rowids_from_text(name_list)
    dscore_list = sorted(zip(name_score_list, dnid_list), reverse=True)

    name_ranks = []
    for rank, (dscore, dnid) in enumerate(dscore_list):
        if dnid == qnid:
            name_ranks.append(rank)

    return annot_ranks, name_ranks


def rank_min_avg(rank_dict, max_rank):
    failed_min_aid_dict = {}
    min_x_list, min_y_list = [], []
    avg_x_list, avg_y_list = [], []
    for rank in range(max_rank):
        failed_min_aid_dict[rank] = []
        count_min, count_avg, total = 0.0, 0.0, 0.0
        for qaid in rank_dict:
            annot_ranks = rank_dict[qaid]
            if len(annot_ranks) == 0:
                annot_ranks = [max_rank + 1]
                # continue
            annot_min_rank = min(annot_ranks)
            annot_avg_rank = sum(annot_ranks) / len(annot_ranks)
            if annot_min_rank <= rank:
                count_min += 1
            else:
                failed_min_aid_dict[rank].append(qaid)

            if annot_avg_rank <= rank:
                count_avg += 1
            total += 1
        percentage_min = count_min / total
        min_x_list.append(rank + 1)
        min_y_list.append(percentage_min)
        percentage_avg = count_avg / total
        avg_x_list.append(rank + 1)
        avg_y_list.append(percentage_avg)

    min_vals = min_x_list, min_y_list
    avg_vals = avg_x_list, avg_y_list

    return min_vals, avg_vals, failed_min_aid_dict


# nids = ibs.get_valid_nids()
# global_aid_list = ut.flatten(ibs.get_name_aids(nids))

global_aid_list = ibs.get_imageset_aids(4)

wanted_species_set = {
    'lynx',
}

query_config_dict_dict = {
    'HotSpotter'           : (32, {}, ),
    'HotSpotter (SV=off)'  : (32, {'sv_on': False}, ),
    'HotSpotter (K=3)'     : (32, {'K': 3}, ),
    'HotSpotter (K=5)'     : (32, {'K': 5}, ),
    'HotSpotter (K=7)'     : (32, {'K': 7}, ),
    'HotSpotter (Knorm=3)' : (32, {'Knorm': 3}, ),
    'HotSpotter (bg=off)'  : (32, {'fg_on': False}, ),
}

random.seed(SEED)

# Load any pre-computed ranks
rank_dict_filepath = join(ibs.dbdir, 'ranks.%d.pkl' % (SEED, ))
# ut.delete(rank_dict_filepath)
if exists(rank_dict_filepath):
    rank_dict = ut.load_cPkl(rank_dict_filepath)
else:
    rank_dict = {}

min_datetime_str = '1990/01/01 00:00:00'
min_datetime_fmtstr = '%Y/%m/%d %H:%M:%S'
min_datetime = datetime.datetime.strptime(min_datetime_str, min_datetime_fmtstr)
# min_unixtime = int(ut.datetime_to_posixtime(min_datetime))
min_unixtime = -1

all_aid_list = sorted(set(global_aid_list))
all_species_list = ibs.get_annot_species(all_aid_list)
all_viewpoint_list = ibs.get_annot_viewpoints(all_aid_list)
all_viewpoint_dict = {}
for all_aid, all_species, all_viewpoint in zip(all_aid_list, all_species_list, all_viewpoint_list):
    if all_species not in wanted_species_set:
        continue
    if all_viewpoint is None:
        continue
    if all_viewpoint not in all_viewpoint_dict:
        all_viewpoint_dict[all_viewpoint] = []
    all_viewpoint_dict[all_viewpoint].append(all_aid)

print(list(map(len, all_viewpoint_dict.values())))
for all_viewpoint in all_viewpoint_dict:
    print(all_viewpoint)
    viewpoint_aid_list_ = all_viewpoint_dict[all_viewpoint]

    viewpoint_nid_list_ = ibs.get_annot_nids(viewpoint_aid_list_)
    viewpoint_nid_dict = {}
    for viewpoint_aid, viewpoint_nid in zip(viewpoint_aid_list_, viewpoint_nid_list_):
        if viewpoint_nid not in viewpoint_nid_dict:
            viewpoint_nid_dict[viewpoint_nid] = []
        viewpoint_nid_dict[viewpoint_nid].append(viewpoint_aid)

    viewpoint_aid_list = []
    for viewpoint_nid in viewpoint_nid_dict:
        aid_list = viewpoint_nid_dict[viewpoint_nid]

        unixtime_list = ibs.get_annot_image_unixtimes(aid_list)

        valid_list = list(np.array(unixtime_list) >= min_unixtime)
        aid_list = ut.compress(aid_list, valid_list)
        unixtime_list = ut.compress(unixtime_list, valid_list)

        values = sorted(zip(unixtime_list, aid_list))
        unixtime_list = ut.take_column(values, 0)
        aid_list = ut.take_column(values, 1)

        # if MIN_TIME_DELTA is not None:
        #     print('\nStarting with:\n\tRowids:  %r\n\tTimes: %r' % (aid_list, unixtime_list))

        #     delta_list = encounter_deltas(unixtime_list)
        #     candidate_list = delta_list < MIN_TIME_DELTA
        #     while True in candidate_list:
        #         candidate_index = np.argmin(delta_list)
        #         # print('Popping index %d' % (candidate_index, ))
        #         unixtime_list.pop(candidate_index)
        #         aid_list.pop(candidate_index)
        #         delta_list = encounter_deltas(unixtime_list)
        #         candidate_list = delta_list < MIN_TIME_DELTA

        #     print('Ended with:\n\tRowids:  %r\n\tTimes: %r' % (aid_list, unixtime_list))

        random.shuffle(aid_list)
        num_aids = len(aid_list)
        if num_aids < MIN_AIDS_PER_NAME:
            # print('WARNING: %r (%d)' % (viewpoint_nid, num_aids, ))
            continue

        keep = min(MAX_AIDS_PER_NAME, len(aid_list))
        aid_list = aid_list[:keep]

        viewpoint_aid_list += aid_list
    viewpoint_aid_list = sorted(list(set(viewpoint_aid_list)))
    viewpoint_nid_list = list(set(ibs.get_annot_nids(viewpoint_aid_list)))
    print('\tChecking %d annotations for %d names' % (len(set(viewpoint_aid_list_)), len(set(viewpoint_nid_list_)), ))
    print('\tUsing    %d annotations for %d names' % (len(viewpoint_aid_list), len(viewpoint_nid_list), ))
    viewpoint_unixtime_list = ibs.get_annot_image_unixtimes(viewpoint_aid_list)
    print('\tMin date: %s' % (ut.unixtime_to_datetimestr(min(viewpoint_unixtime_list)), ))
    print('\tMax date: %s' % (ut.unixtime_to_datetimestr(max(viewpoint_unixtime_list)), ))
    all_viewpoint_dict[all_viewpoint] = sorted(viewpoint_aid_list)

print(list(map(len, all_viewpoint_dict.values())))

for query_config_label in rank_dict:
    for key in rank_dict[query_config_label]:
        print('%s %s %d' % (query_config_label, key, len(rank_dict[query_config_label][key]), ))

for aid_viewpoint in all_viewpoint_dict:
    viewpoint_aid_list = all_viewpoint_dict[aid_viewpoint]
    viewpoint_aid_list = list(set(viewpoint_aid_list))

    for query_config_label in query_config_dict_dict:
        print('Processing %r' % (query_config_label, ))

        if query_config_label not in rank_dict:
            rank_dict[query_config_label] = {
                'annots': {},
                'names': {},
            }

        chunksize, query_config_dict = query_config_dict_dict[query_config_label]

        annots = rank_dict[query_config_label]['annots'].keys()
        names = rank_dict[query_config_label]['names'].keys()
        done_aids = list(set(annots) & set(names))

        if ALL_VS_ALL:
            qaid_list = sorted(list(set(viewpoint_aid_list) - set(done_aids)))
            daid_list = viewpoint_aid_list
        else:
            qaid_list = sorted(list(set(viewpoint_aid_list) - set(done_aids)))
            daid_list = global_aid_list

        print('\tmissing qaids = %d' % (len(qaid_list), ))
        if len(qaid_list) == 0:
            continue

        qaid_chunks = list(ut.ichunks(qaid_list, chunksize))
        for qaid_chunk in tqdm.tqdm(qaid_chunks, desc=query_config_label):
            try:
                query_results = ibs.query_chips_graph(
                    qaid_list=qaid_chunk,
                    daid_list=daid_list,
                    query_config_dict=query_config_dict,
                    echo_query_params=False,
                    cache_images=False,
                    n=0,
                )
            except Exception:
                query_results = None

            if query_results is not None:
                cm_dict = query_results['cm_dict']
                cm_keys = list(cm_dict.keys())
                for cm_key in cm_keys:
                    cm = cm_dict[cm_key]
                    qannot_uuid = cm['qannot_uuid']
                    qaid = ibs.get_annot_aids_from_uuid(qannot_uuid)
                    annot_ranks, name_ranks = rank(ibs, query_results, cm_key=cm_key)
                    rank_dict[query_config_label]['annots'][qaid] = annot_ranks
                    rank_dict[query_config_label]['names'][qaid] = name_ranks

                ut.save_cPkl(rank_dict_filepath, rank_dict)

        ut.save_cPkl(rank_dict_filepath + '.backup', rank_dict)


####################


def get_marker(index, total):
    marker_list = ['o', 'X', '+', '*']
    num_markers = len(marker_list)
    if total <= 4:
        index_ = 0
    else:
        index_ = index % num_markers
    marker = marker_list[index_]
    return marker

for viewpoint in all_viewpoint_dict:
    print('Processing %s' % (viewpoint, ))
    viewpoint_aid_set = set(all_viewpoint_dict[viewpoint])

    datapoints = {
        'annot': {},
        'name': {},
    }

    rank_dict_ = {}
    qaid_set = set([])
    for query_config_label in rank_dict:
        annot_ranks = rank_dict[query_config_label]['annots']
        name_ranks = rank_dict[query_config_label]['names']

        annot_ranks_ = {}
        for qaid in annot_ranks:
            if qaid in viewpoint_aid_set:
                annot_ranks_[qaid] = annot_ranks[qaid]
                qaid_set.add(qaid)

        name_ranks_ = {}
        for qaid in name_ranks:
            if qaid in viewpoint_aid_set:
                name_ranks_[qaid] = name_ranks[qaid]
                qaid_set.add(qaid)

        rank_dict_[query_config_label] = {}
        rank_dict_[query_config_label]['annots'] = annot_ranks_
        rank_dict_[query_config_label]['names'] = name_ranks_

    qaid_list_ = list(qaid_set)
    qnid_list_ = ibs.get_annot_nids(qaid_list_)

    fig_ = plt.figure(figsize=(26, 10), dpi=300)  # NOQA
    fig_.subplots_adjust(top=0.83)

    viewpoints_ = ut.dict_hist(ibs.get_annot_viewpoints(qaid_list_))
    if ALL_VS_ALL:
        query_type_str = 'All vs. All Query'
    else:
        request_qaid_list = sorted(list(set(viewpoint_aid_list)))
        request_daid_list = sorted(list(set(global_aid_list)))
        request_args = (len(request_qaid_list), len(request_daid_list), )
        query_type_str = 'Query (%d) vs. Database (%d) Query' % request_args

    args = (
        query_type_str,
        len(set(qaid_list_)),
        len(set(qnid_list_)),
        MIN_AIDS_PER_NAME,
        '∞' if np.isinf(MAX_AIDS_PER_NAME) else '%d' % (MAX_AIDS_PER_NAME, ),
        '∞' if MIN_TIME_DELTA is None else '%d' % (MIN_TIME_DELTA, ),
    )
    plot_title = 'Lynx - %s - %d Annots, %d Names [Filters: Min %d, Max %s, Encounter %s sec.]' % args
    fig_.suptitle(plot_title, size=18)

    wanted_dict_keys = list(query_config_dict_dict.keys())

    values_list = []
    failed_min_aid_dict_list = []
    for query_config_label in rank_dict_:
        if query_config_label not in wanted_dict_keys:
            continue
        annot_ranks = rank_dict_[query_config_label]['annots']
        min_vals, avg_vals, failed_min_aid_dict = rank_min_avg(annot_ranks, MAX_RANK)
        failed_min_aid_dict_list.append(failed_min_aid_dict)
        min_x_list, min_y_list = min_vals
        # avg_x_list, avg_y_list = avg_vals
        args = (
            query_config_label,
            min_y_list[0],
            min_y_list[4],
            MAX_RANK,
            min_y_list[-1],
        )
        query_config_label_ = '%s [1:%0.03f, 5:%0.03f, %d:%0.03f]' % args
        values_list.append(
            (
                query_config_label,
                query_config_label_,
                min_x_list,
                min_y_list,
            )
        )
        datapoints['annot'][query_config_label] = min_y_list

    color_list = []
    color_list += pt.distinct_colors(len(values_list), randomize=False)

    axes_ = plt.subplot(121)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_ylabel('Percent Matched Correct Name')
    axes_.set_xlabel('Rank (Matched Within Top-K Results) - 90% Threshold Highlighted in Gold')
    axes_.set_xticks(np.arange(1, MAX_RANK + 1, step=1))
    axes_.set_yticks(np.arange(0.0, 1.1, step=0.1))
    axes_.set_xlim([0.75, MAX_RANK + 0.25])
    axes_.set_ylim([0.0, 1.0])

    good_x = None
    for values in values_list:
        x_list, y_list = values[-2:]
        for x, y in zip(x_list, y_list):
            if y >= 0.50:
                if good_x is None:
                    good_x = x
                else:
                    good_x = min(good_x, x)

    plt.plot([1.0 + 0.05, 1.0 + 0.05], [0.0, 1.0], color='dimgray', linestyle='-', linewidth=1)
    plt.plot([5.0, 5.0], [0.0, 1.0], color='dimgray', linestyle='-', linewidth=1)
    plt.plot([MAX_RANK - 0.05, MAX_RANK - 0.05], [0.0, 1.0], color='dimgray', linestyle='-', linewidth=1)

    if good_x is not None:
        plt.plot([good_x, good_x], [0.0, 1.0], color='gold', linestyle='-', linewidth=2)

    good_annot_x = int(good_x) - 1
    failed_aids = failed_min_aid_dict_list[-1][good_annot_x]

    zipped = list(zip(color_list, values_list))
    total = len(zipped)
    for index, (color, values) in enumerate(zipped):
        label, label_, x_list, y_list = values
        marker = get_marker(index, total)
        plt.plot(x_list, y_list, color=color, marker=marker, label=label_, alpha=1.0)

    plt.title('One-to-Many Annotations - Cumulative Match Rank')
    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, .102), loc=3, ncol=2, mode='expand',
               borderaxespad=0.0)

    values_list = []
    for query_config_label in rank_dict_:
        if query_config_label not in wanted_dict_keys:
            continue
        name_ranks = rank_dict_[query_config_label]['names']
        min_vals, avg_vals, _ = rank_min_avg(name_ranks, MAX_RANK)
        min_x_list, min_y_list = min_vals
        # avg_x_list, avg_y_list = avg_vals
        args = (
            query_config_label,
            min_y_list[0],
            min_y_list[4],
            MAX_RANK,
            min_y_list[-1],
        )
        query_config_label_ = '%s [1:%0.03f, 5:%0.03f, %d:%0.03f]' % args
        values_list.append(
            (
                query_config_label,
                query_config_label_,
                min_x_list,
                min_y_list,
            )
        )
        datapoints['name'][query_config_label] = min_y_list

    color_list = []
    color_list += pt.distinct_colors(len(values_list), randomize=False)

    axes_ = plt.subplot(122)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_ylabel('Percent Matched Correct Name')
    axes_.set_xlabel('Rank (Matched Within Top-K Results) - 90% Threshold Highlighted in Gold')
    axes_.set_xticks(np.arange(1, MAX_RANK + 1, step=1))
    axes_.set_yticks(np.arange(0.0, 1.1, step=0.1))
    axes_.set_xlim([0.75, MAX_RANK + 0.25])
    axes_.set_ylim([0.0, 1.0])

    good_x = None
    for values in values_list:
        x_list, y_list = values[-2:]
        for x, y in zip(x_list, y_list):
            if y >= 0.90:
                if good_x is None:
                    good_x = x
                else:
                    good_x = min(good_x, x)

    plt.plot([1.0 + 0.05, 1.0 + 0.05], [0.0, 1.0], color='dimgray', linestyle='-', linewidth=1)
    plt.plot([5.0, 5.0], [0.0, 1.0], color='dimgray', linestyle='-', linewidth=1)
    plt.plot([MAX_RANK - 0.05, MAX_RANK - 0.05], [0.0, 1.0], color='dimgray', linestyle='-', linewidth=1)

    if good_x is not None:
        plt.plot([good_x, good_x], [0.0, 1.0], color='gold', linestyle='-', linewidth=2)

    zipped = list(zip(color_list, values_list))
    total = len(zipped)
    for index, (color, values) in enumerate(zipped):
        label, label_, x_list, y_list = values
        marker = get_marker(index, total)
        plt.plot(x_list, y_list, color=color, marker=marker, label=label_, alpha=1.0)

    plt.title('One-to-Many Names - Cumulative Match Rank')
    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, .102), loc=3, ncol=2, mode='expand',
               borderaxespad=0.0)

    fig_filename = 'matching-ybt-cmc-hotspotter-%s-max-%s-seed-%d.png' % (viewpoint, MAX_AIDS_PER_NAME, SEED, )
    fig_path = abspath(join('/', 'data', 'db', fig_filename))
    plt.savefig(fig_path, bbox_inches='tight')
