def parse_volume(volume_str):
    return [[float(volume_str)]]

def parse_pd_notation(pd_str):
    pd_str = pd_str.replace(' ', '')
    outer_brackets_removed = pd_str[2:-2]
    splitted_by_lists = outer_brackets_removed.split('];[')
    pd_notation = []
    for sublist_str in splitted_by_lists:
        numbers_strs = sublist_str.split(';')
        sublist = []
        for num_str in numbers_strs:
            sublist.append(int(num_str))
        pd_notation.append(sublist)
    return pd_notation

def parse_list_of_features(list_of_feature_str, parse_function):
    parsed_features = []
    for feature_str in list_of_feature_str:
        parsed_feature = parse_function(feature_str)
        parsed_features.append(parsed_feature)
    return parsed_features