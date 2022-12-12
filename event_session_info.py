import json

### CREATE EVENT LIST ###
event_list = {
     'Image/Inanimate/Natural/Orange': 1,
     'Image/Inanimate/Artificial/Bench': 2,
     'Image/Inanimate/Artificial/Remote': 3,
     'Image/Inanimate/Artificial/Car': 4,
     'Image/Inanimate/Artificial/Stove': 5,
     'Image/Animate/Coolguy': 6,
     'Image/Inanimate/Artificial/Table': 7,
     'Image/Inanimate/Natural/Apple': 8,
     'Image/Inanimate/Artificial/Cart': 9,
     'Image/Animate/Natural/Dog': 10,
     'Image/Animate/Natural/Fox': 11,
     'Image/Inanimate/Artificial/Bus': 12,
     'Image/Inanimate/Artificial/Train': 13,
     'Image/Inanimate/Artificial/Ipod': 14,
     'Image/Inanimate/Artificial/Pizza': 15,
     'Image/Animate/Natural/Bird': 16,
     'Image/Animate/Natural/Horse': 17,
     'Image/Inanimate/Artificial/Laptop': 18,
     'Image/Animate/Natural/Polarbear': 19,
     'Image/Inanimate/Artificial/Basketball': 20,
     'Image/Inanimate/Artificial/Piano': 21,
     'Image/Inanimate/Artificial/Acousticguitar': 22,
     'Image/Inanimate/Artificial/Baseball': 23,
     'Image/Animate/Natural/Seal': 24,
     'Image/Inanimate/Artificial/Chair': 25,
     'Image/Animate/Natural/Orangutang': 26,
     'Image/Inanimate/Artificial/Bowl': 27,
     'Image/Animate/Natural/Tiger': 28,
     'Image/Inanimate/Artificial/Scooter': 29,
     'Image/Inanimate/Artificial/Tie': 30,
     'Image/Inanimate/Artificial/Printer': 31,
     'Image/Animate/Natural/Lion': 32,
     'Image/Inanimate/Artificial/Nail': 33,
     'Image/Inanimate/Artificial/Drum': 34,
     'Image/Inanimate/Artificial/Bow': 35,
     'Image/Inanimate/Natural/Fig': 36,
     'Image/Animate/Natural/Butterfly': 37,
     'Image/Inanimate/Artificial/Lamp': 38,
     'Image/Inanimate/Natural/Banana': 39,
     'Image/Inanimate/Artificial/Couch': 40,
     'Image/Inanimate/Natural/Lemon': 41,
     'Image/Inanimate/Artificial/Vacuum': 42,
     'Image/Inanimate/Artificial/Hammer': 43,
     'Image/Inanimate/Artificial/Sunglasses': 44, 
     'Image/Inanimate/Artificial/Bicycle': 45,
     'Image/Animate/Natural/Hare': 46,
     'Image/Inanimate/Artificial/Measuringcup': 47,
     'Image/Animate/Natural/Elephant': 48,
     'Image/Inanimate/Artificial/Microwave': 49,
     'Image/Inanimate/Artificial/Volleyball': 50,
     'Image/Inanimate/Natural/Strawberry': 51,
     'Image/Animate/Natural/Sheep': 52,
     'Image/Animate/Natural/Frog': 53,
     'Image/Inanimate/Artificial/Washingmachine': 54,
     'Image/Inanimate/Artificial/Fridge': 55,
     'Image/Animate/Natural/Turtle': 56,
     'Image/Inanimate/Artificial/Ax': 57,
     'Image/Inanimate/Artificial/Helmet': 58,
     'Image/Animate/Natural/Camel': 59,
     'Image/Inanimate/Artificial/Lipstick': 60,
     'Image/Inanimate/Artificial/Airplane': 61,
     'Image/Inanimate/Artificial/Dishwasher': 62,
     'Image/Inanimate/Artificial/Burger': 63,
     'Image/Inanimate/Artificial/Backpack': 64,
     'Image/Inanimate/Artificial/Handbag': 65,
     'Image/Animate/Natural/Hamster': 66,
     'Image/Inanimate/Artificial/Microphone': 67,
     'Image/Inanimate/Natural/Mushroom': 68,
     'Image/Animate/Natural/Cow': 69,
     'Image/Inanimate/Artificial/Violin': 70,
     'Image/Animate/Natural/Killerwhale': 71,
     'Image/Inanimate/Natural/Cucumber': 72,
     'Image/Inanimate/Artificial/Ruler': 73,
     'Image/Inanimate/Natural/Pineapple': 74,
     'Image/Inanimate/Artificial/Harp': 75,
     'Image/Animate/Natural/Squrriel': 76, 
     'Image/Inanimate/Artificial/Ringbinder': 77,
     'Image/Inanimate/Artificial/Trumpet': 78,
     'Image/Inanimate/Artificial/Plasticbag': 79,
     'Image/Inanimate/Artificial/Carafe': 80,
     'Image/Animate/Natural/Snail':81,
     'Image/Inanimate/Artificial/Toaster': 82,
     'Image/Inanimate/Artificial/Banjo': 83,
     'Image/Inanimate/Artificial/Skrewdriver': 84,
     'Image/Inanimate/Artificial/Snowscooter': 85,
     'Image/Animate/Natural/Pig': 86,
     'Image/Inanimate/Natural/Pomegranate': 87,
     'Image/Inanimate/Artificial/Harmonica': 88,
     'Image/Inanimate/Artificial/Rachet': 89,
     'Image/Inanimate/Artificial/Bagel': 90,
     'Image/Inanimate/Artificial/Trumpet': 91,
     'Image/Inanimate/Artificial/Syringe': 92,
     'Image/Animate/Natural/Goldfish': 93,
     'Image/Inanimate/Artificial/Hairdryer': 94,
     'Image/Animate/Natural/Seastar': 95,
     'Image/Inanimate/Artificial/Hotdog': 96,
     'Image/Animate/Natural/Ladybug': 97,
     'Image/Inanimate/Artificial/Ship': 98,
     'Image/Animate/Natural/Jellyfish': 99,
     'Image/Inanimate/Artificial/Coffeemachine': 100,
     'Image/Inanimate/Artificial/Computerscreen': 101,
     'Image/Inanimate/Artificial/Pretzel': 102,
     'Image/Inanimate/Natural/Plant': 103, 
     'Image/Inanimate/Artificial/Golfball': 104,
     'Image/Inanimate/Artificial/Alarmclock': 105,
     'Image/Inanimate/Artificial/Traficlight': 106,
     'Image/Inanimate/Artificial/Wine': 107,
     'Image/Inanimate/Artificial/Pan': 108,
     'Image/Inanimate/Artificial/Handweight': 109,
     'Image/Inanimate/Artificial/Football': 110,
     'Image/Inanimate/Natural/Paprika': 111,
     'Image/Inanimate/Artificial/Bottleopener': 112,
     'Image/Inanimate/Artificial/Tennisball': 113,
     'Image/Inanimate/Artificial/Computermouse': 114,
     'Image/Inanimate/Artificial/Mug': 115,
     'Image/Inanimate/Artificial/Keyboard': 116,
     'Image/Inanimate/Artificial/Canopener': 117,
     'Image/Inanimate/Artificial/Fan': 118

} 

print(event_list)
with open('event_ids.txt', 'w') as f:
     f.write(json.dumps(event_list))


### CREATE SESSION INFO ###
# Bad channels found by visual inspection of MEG signal
# Noise components determined through visual inspection of spatial map, time course and spectrum (in notebook check_ica)
file_list = {
     'memory_01.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2513', 'MEG2613', 'MEG1341', 'MEG1831'], 'date': 'xxxxx', 'noise_components': [0, 15, 17], 'tmin': 7, 'tmax': 1537},
     'memory_02.fif': {'bad_channels': ['MEG0422', 'MEG0413', 'MEG1423', 'MEG2533'], 'noise_components':[0, 18, 20], 'tmin':55, 'tmax': 1559},
     'memory_03.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG0921', 'MEG0811'], 'noise_components': [1,11,12], 'tmin': 9, 'tmax': 743.5},
     'memory_04.fif': {'bad_channels': ['MEG1222', 'MEG0422', 'MEG1423', 'MEG2613', 'MEG0811', 'MEG0921', 'MEG2413'], 'noise_components': [4, 8, 10], 'tmin':8, 'tmax':758},
     'memory_05.fif': {'bad_channels': ['MEG0821', 'MEG0411', 'MEG2613', 'MEG1612', 'MEG1423', 'MEG0933', 'MEG0422'],'noise_components': [1, 17], 'tmin':13, 'tmax':376}, # cut of the last repetitions -> very noisy
     'memory_06.fif': {'bad_channels': ['MEG2141', 'MEG0821', 'MEG0411', 'MEG2613', 'MEG2413', 'MEG1612', 'MEG1423', 'MEG0933', 'MEG0422'], 'noise_components': [6, 27, 30], 'tmin': 10, 'tmax': 773},
     'memory_07.fif': {'bad_channels': ['MEG0422', 'MEG0632', 'MEG1423', 'MEG1731', 'MEG1331'], 'noise_components': [4, 15, 18], 'tmin':15, 'tmax': 769},
     'memory_08.fif': {'bad_channels': ['MEG0422', 'MEG0632', 'MEG1423', 'MEG2513', 'MEG1731'],'noise_components': [0, 13, 15, 17], 'tmin': 55, 'tmax':423},
     'memory_09.fif': {'bad_channels': ['MEG0422', 'MEG0632', 'MEG1423', 'MEG0341', 'MEG2631'], 'noise_components': [10, 15, 27], 'tmin': 11, 'tmax': 749},
     'memory_10.fif': {'bad_channels': ['MEG0422', 'MEG0632', 'MEG1423', 'MEG0341', 'MEG2631'], 'noise_components': [13, 14, 0], 'tmin':18,  'tmax':386},
     'memory_11.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2542', 'MEG0531', 'MEG1211'], 'noise_components': [6, 17, 20], 'tmin': 21, 'tmax': 775},
     'memory_12.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2513', 'MEG0411'], 'tmin': 26, 'noise_components': [0, 10, 11], 'tmax': 785},
     'memory_13.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2513', 'MEG1331'], 'tmin': 12, 'noise_components': [0, 11, 12], 'tmax': 377},
     'memory_14.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2532', 'MEG2613', 'MEG2331'], 'noise_components': [7, 10, 16], 'tmin': 10, 'tmax': 756},
     'memory_15.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2532', 'MEG2331'], 'noise_components': [5, 6, 7], 'tmin': 11, 'tmax': 379},
     'visual_03.fif': {'bad_channels': ['MEG0422'], 'noise_components':  [1, 13, 18], 'tmin': 40, 'tmax':1284},
     'visual_04.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2531', 'MEG1341', 'MEG1831', 'MEG2542'], 'noise_components': [1, 14, 15,17], 'tmin': 10, 'tmax':1151},
     'visual_05.fif': {'bad_channels': ['MEG0422', 'MEG0933', 'MEG1122', 'MEG1423', 'MEG2513'], 'noise_components': [2, 9, 10, 153], 'tmin': 24, 'tmax': 775},
     'visual_06.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2613'], 'noise_components':[1, 11, 14],'tmin': 35, 'tmax':792},
     'visual_07.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2422'], 'noise_components':[5, 13, 23],'tmin': 20, 'tmax':760},
     'visual_08.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2513', 'MEG0441', 'MEG1831', 'MEG0441'],'noise_components': [2, 21], 'tmin': 14, 'tmax':1536, 'note': 'TOTAL CHAOS FROM (767 TO 776) AND (1152 TO 1170)'},
     'visual_09.fif': {'bad_channels': ['MEG0422', 'MEG2613', 'MEG1423'], 'noise_components': [3, 16], 'tmin': 7, 'tmax': 758},
     'visual_10.fif': {'bad_channels': ['MEG0422', 'MEG1423', 'MEG2513', 'MEG2613'], 'noise_components': [14, 23, 25], 'tmin': 13, 'tmax':760},
     'visual_11.fif': {'bad_channels': ['MEG0422', 'MEG2613', 'MEG1423', 'MEG2513'], 'noise_components':[1, 9, 15], 'tmin':12, 'tmax':1147},
     'visual_12.fif': {'bad_channels': ['MEG0422',  'MEG1423', 'MEG2513', 'MEG2613'], 'noise_components': [8, 24], 'tmin':12, 'tmax':789},
     'visual_13.fif': {'bad_channels': ['MEG0422', 'MEG0542', 'MEG1423', 'MEG2613', 'MEG0811'], 'noise_components': [6, 13, 8], 'tmin':9, 'tmax':764},
     'visual_14.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG1133', 'MEG1323', 'MEG1423', 'MEG2513', 'MEG1211', 'MEG2631'], 'noise_components': [6,29 ], 'tmin':14, 'tmax':374},
     'visual_15.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG0722', 'MEG1133', 'MEG1323', 'MEG1423', 'MEG2623', 'MEG0121', 'MEG1211'], 'noise_components': [9, 22], 'tmin':15, 'tmax':381},
     'visual_16.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG0722', 'MEG1133', 'MEG1323', 'MEG1423', 'MEG2513', 'MEG0411', 'MEG1211', 'MEG2111'], 'noise_components': [8, 88], 'tmin':12, 'tmax':377},
     'visual_17.fif': {'bad_channels': ['MEG0242', 'MEG0422', 'MEG1423', 'MEG2513', 'MEG2613', 'MEG0821', 'MEG1511', 'MEG0243', 'MEG0722'], 'noise_components': [14, 31], 'tmin':9, 'tmax':375},
     'visual_18.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG0722', 'MEG1423', 'MEG2613', 'MEG0821', 'MEG1511'], 'noise_components': [5, 38], 'tmin':20, 'tmax':386},
     'visual_19.fif': {'bad_channels': ['MEG1511', 'MEG0821', 'MEG2613', 'MEG2513', 'MEG1423', 'MEG0422', 'MEG0243', 'MEG0242', 'MEG0722'], 'noise_components': [6, 15, 16], 'tmin':9, 'tmax':376},
     'visual_20.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG0723', 'MEG1423', 'MEG1742', 'MEG2413', 'MEG2613'], 'noise_components': [11,34], 'tmin':13, 'tmax':379},
     'visual_21.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG0723', 'MEG1423', 'MEG2413', 'MEG2613', 'MEG1511', 'MEG1531'], 'noise_components': [17, 30], 'tmin':20, 'tmax':381},
     'visual_22.fif': {'bad_channels': ['MEG0242', 'MEG0243', 'MEG0422', 'MEG1423', 'MEG1742', 'MEG2413', 'MEG2613', 'MEG0121', 'MEG1531', 'MEG1511', 'MEG0132'], 'noise_components': [16, 27], 'tmin':11, 'tmax':379},
     'visual_23.fif': {'bad_channels': ['MEG0243', 'MEG0412', 'MEG0422', 'MEG1423', 'MEG0111', 'MEG0121', 'MEG1331', 'MEG0132'], 'noise_components': [5, 20], 'tmin':12, 'tmax':378},
     'visual_24.fif': {'bad_channels': ['MEG0132', 'MEG0412', 'MEG0422', 'MEG2532', 'MEG0111', 'MEG1331'], 'noise_components': [12, 18], 'tmin':15, 'tmax':380},
     'visual_25.fif': {'bad_channels': ['MEG0132', 'MEG0422', 'MEG0412', 'MEG2532', 'MEG2531'], 'noise_components': [4, 14], 'tmin':9, 'tmax':352},
     'visual_26.fif': {'bad_channels': ['MEG2631', 'MEG0412', 'MEG0422', 'MEG1122', 'MEG0111', 'MEG0821'], 'noise_components': [2, 7], 'tmin':8, 'tmax':376},
     'visual_27.fif': {'bad_channels': ['MEG0422', 'MEG1223', 'MEG2613', 'MEG0411', 'MEG1431'], 'noise_components': [14, 19, 24], 'tmin':13, 'tmax':380},
     'visual_28.fif': {'bad_channels': ['MEG0343', 'MEG0422', 'MEG0923', 'MEG1223', 'MEG2613', 'MEG1941', 'MEG2631'], 'noise_components': [8, 11, 12], 'tmin':8, 'tmax':374},
     'visual_29.fif': {'bad_channels': ['MEG0343', 'MEG0422', 'MEG1223', 'MEG2613', 'MEG2631'], 'noise_components': [3, 15, 17], 'tmin':8, 'tmax':375},
     'visual_30.fif': {'bad_channels': ['MEG0232', 'MEG0422', 'MEG0722', 'MEG0121', 'MEG1341','MEG2141'], 'noise_components': [10], 'tmin':19, 'tmax':387},
     'visual_31.fif': {'bad_channels': ['MEG0232', 'MEG0422', 'MEG0722', 'MEG0121', 'MEG1341', 'MEG2141'], 'noise_components': [3, 8], 'tmin':10, 'tmax':378},
     'visual_32.fif': {'bad_channels': ['MEG0232', 'MEG0422', 'MEG0722', 'MEG1341', 'MEG2141'], 'noise_components': [8, 4], 'tmin':15, 'tmax':378},
     'visual_33.fif': {'bad_channels': ['MEG0343', 'MEG0422', 'MEG0723', 'MEG2613', 'MEG1513', 'MEG1411'], 'noise_components': [13, 25], 'tmin':11, 'tmax':368},
     'visual_34.fif': {'bad_channels': ['MEG0422', 'MEG0723', 'MEG2613'], 'noise_components': [16], 'tmin':10, 'tmax':377},
     'visual_35.fif': {'bad_channels': ['MEG0422', 'MEG0723', 'MEG2213', 'MEG2613', 'MEG2221'], 'noise_components': [13, 23], 'tmin':9, 'tmax':378},
     'visual_36.fif': {'bad_channels': ['MEG0422', 'MEG0723', 'MEG0922', 'MEG2613', 'MEG0121', 'MEG1011', 'MEG2141', 'MEG2612'], 'noise_components': [25, 36], 'tmin':10, 'tmax':377},
     'visual_37.fif': {'bad_channels': ['MEG0422', 'MEG0723', 'MEG2613', 'MEG0121', 'MEG1011'], 'noise_components': [25, 35], 'tmin':15, 'tmax':381},
     'visual_38.fif': {'bad_channels': ['MEG0422', 'MEG0723', 'MEG2612', 'MEG1011'], 'noise_components': [11, 17, 26], 'tmin':10, 'tmax':377},
     'rest_01.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_02.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_03.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_04.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_05.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_06.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_belt_01.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_mouthblock_01.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]},
     'rest_noseclip_01.fif': {'bad_channels': [], 'noise_components': [], 'tmin':[], 'tmax':[]}


} 

print(file_list)
with open('session_info.txt', 'w') as f:
     f.write(json.dumps(file_list))