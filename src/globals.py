def activity_dict():
    act_dict = {
        'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
        'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'), 'act06': (5, 'riding escalator up'),
        'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'),
        'act10': (9, 'drinking'),
        'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'),
        'act14': (13, 'reading'),
        'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'),
        'act18': (17, 'push-ups'),
        'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')
    }
    return act_dict


def sensor_columns():
    all_sns = ['accx', 'accy', 'accz',
               'grax', 'gray', 'graz',
               'gyrx', 'gyry', 'gyrz',
               'lacx', 'lacy', 'lacz',
               'magx', 'magy', 'magz',
               'rotx', 'roty', 'rotz', 'rote']
    return all_sns
