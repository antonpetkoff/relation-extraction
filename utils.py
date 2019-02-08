def update_dict(params, keys, value):
        if len(keys) == 1:
            params[keys[0]] = value
        else:
            update_dict(params[keys[0]], keys[1:], value)

def flatten(list):
    return [item for sublist in list for item in sublist]
