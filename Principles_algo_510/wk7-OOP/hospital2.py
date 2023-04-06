#!/usr/bin/env python3

from temperature2 import fahrenheit_to_celsius

temp_f = 98.6
temp_c = fahrenheit_to_celsius(temp_f)
print(f'{temp_f}F = {fahrenheit_to_celsius(float(temp_f))}C')

