#!/usr/bin/env python3

# Based on https://www.youtube.com/watch?v=f26nAmfJggw


# import Planet class from planet module in space package

from space.planet import Planet 

# import functions planet_mass and planet_volume from calc module in space package

from space.calc import planet_mass, planet_volume


naboo = Planet('Naboo', 300000, 8, 'Naboo System')

naboo_mass = planet_mass(naboo.gravity, naboo.radius)
naboo_volume = planet_volume(naboo.radius)

print(f'{naboo.name}: \n - mass: {naboo_mass} \n - volume: {naboo_volume} \n - density: {naboo_mass / naboo_volume} \n - {naboo.spin()}')

print(Planet.commons())

print(naboo.commons())
