#!/usr/bin/env python3

def fahrenheit_to_celsius(temp_f):
    return (temp_f - 32) * 5 / 9

while temp_f_s := input("Enter temperature in Fahrenheit: "):
   temp_c = fahrenheit_to_celsius(float(temp_f_s))
   print(f'{temp_f_s}F = {temp_c:.2f}C')

