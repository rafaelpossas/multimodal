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

def activity_dict_plain():
    act_dict = ['(a) walking', '(b) walking upstairs', '(c) walking downstairs', '(d) riding elevator up', '(e) riding elevator down',
                '(f) riding escalator up', '(g) riding escalator down', '(h) sitting', '(i) eating', '(j) drinking', '(k) texting', '(l) phone calls',
                '(m) working on pc', '(n) reading', '(o) writing sentences', '(p) organizing files', '(q) running', '(r) push-ups', '(s) sit-ups',
                '(t) cycling'
    ]
    return act_dict

def activity_dict_plain_letters():
    act_dict = ['(a)', '(b)', '(c)', '(d)', '(e)',
                '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)',
                '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)',
                '(t)'
    ]
    return act_dict

def sensor_columns():
    all_sns = ['accx', 'accy', 'accz',
               'grax', 'gray', 'graz',
               'gyrx', 'gyry', 'gyrz',
               'lacx', 'lacy', 'lacz',
               'magx', 'magy', 'magz',
               'rotx', 'roty', 'rotz', 'rote']
    return all_sns


def activity_dict_vuzix():
    dict = ['walking', 'climbing stairs', 'chopping food', 'riding elevator', 'brushing teeth',
            'riding escalator', 'talking to people', 'watching tv', 'eating', 'cooking on stove',
            'browsing mobile phone', 'washing dishes', 'working on pc', 'reading', 'writing',
            'lying down', 'running', 'doing push ups', 'doing sit ups', 'cycling']
    return dict
