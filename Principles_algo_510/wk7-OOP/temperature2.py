#!/usr/bin/env python3

import sys

def fahrenheit_to_celsius(temp_f):
    return (temp_f - 32) * 5 / 9

if __name__ == "__main__":
    # sys.argv is the list of arguments; sys.argv[0] is filename
    if len(sys.argv) == 1:
        while temp_f_s := input("Enter temperature in Fahrenheit: "):
            temp_c = fahrenheit_to_celsius(float(temp_f_s))
            print(f'{temp_f_s}F = {temp_c:.2f}C')
    else:
        for temp_f_s in sys.argv[1:]:
            temp_c = fahrenheit_to_celsius(float(temp_f_s))
            print(f'{temp_f_s}F = {temp_c:.2f}C')
else:
    # just print this and provide the function fahrenheit_to_celsius
    print(f'... Importing module {__name__} ...')

