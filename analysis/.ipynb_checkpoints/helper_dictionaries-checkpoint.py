#dictionaries designed to help convert numbers to months, etc...

#dictionary to identify month numbers as a letter, designed to name seasons (e.g. JAS)
#0:D is included to help with modulus math when crossing years, e.g. 12%12 goes to 0
month_number_dict = {
    1: 'J',
    2: 'F',
    3: 'M',
    4: 'A',
    5: 'M',
    6: 'J',
    7: 'J',
    8: 'A',
    9: 'S',
    10:'O',
    11:'N',
    12:'D',
    0:'D'}