import math
import cartopy
import pyomo.environ as pyo
import itertools
import sys
import os
import random
from datetime import date
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import pandas as pd

# Global parameters
BAND_NAME = "Tyler Quick"
BAND_GENRE = "rock"
MAX_TOUR_DAYS = 60
HOME_BASE = "Fayetteville"  # Starting and ending city for the tour
TOUR_BUDGET = 100000          # Starting budget in dollars
MAX_DISTANCE = 5000          # Maximum distance per leg if no fuel-tank limiter 
FUEL_TANK_CAPACITY = 20      # fuel tank capacity (gallons)
FUEL_EFFICIENCY = 25         # miles per gallon
BAND_POPULARITY = 0.9       # Used as a threshold for venue gatekeeping. IRL it could be some calculated metric (eg. Probability a random person knows the band, spotify followers, etc.)
NUM_BAND_MEMBERS = 6         # Total members in the band
AVG_FLIGHT_COST_PER_PERSON = 386  
FLIGHT_DISTANCE_THRESHOLD = 2000  # Miles - if distance > this, flight becomes the only option
MIN_STOPS = 10
INTERNATIONAL = False    # If True, include international cities; if False, only US cities.
GAS_DIST_LIMITER = False     # True: original fuel-tank model; False: distance-only limiter.

# Calculated Parameters

NUM_VEHICLES = np.ceil(NUM_BAND_MEMBERS/5)  # Vehicles required (5 people per vehicle)
ROOMS_NEEDED = math.ceil(NUM_BAND_MEMBERS / 2)
if GAS_DIST_LIMITER:
    MAX_DISTANCE = FUEL_TANK_CAPACITY * FUEL_EFFICIENCY

# ============================================================================
# CLASSES
# ============================================================================
class Venue:
    """
    Represents a music venue with associated revenue information.
    """
    def __init__(self, name, ticket_price, capacity,
                 flat_payment=0, ticket_proportion=0.0, popularity_threshold = 0.0):
        """
        Initialize a venue.

        Parameters:
            name (str): Venue name
            ticket_price (float): Price per ticket in dollars
            capacity (int): Maximum number of tickets available
            flat_payment (float): Flat fee paid to the band
            ticket_proportion (float): Fraction of ticket revenue the band retains (0.0-1.0)
            popularity_threshold (float): Gatekeeps whether or not a band can play at a venue. Assume it is the probability that a random person will have heard of the band
        """
        self.name = name
        self.ticket_price = ticket_price
        self.capacity = capacity
        self.flat_payment = flat_payment
        self.ticket_proportion = ticket_proportion
        self.popularity_threshold = popularity_threshold

    def expected_revenue(self, regional_fanbase_strength):
        """
        Calculate expected revenue from performing at this venue.

        Revenue consists of:
          - Flat payment
          - Ticket revenue (ticket_proportion * tickets_sold * ticket_price)
          - Merchandise sales (20% of crowd buys merch at $35)

        Parameters:
            regional_fanbase_strength (float): Band popularity in the city (0.0-1.0)

        Returns:
            float: Total expected revenue in dollars
        """
        tickets_sold = self.capacity * regional_fanbase_strength * BAND_POPULARITY
        ticket_rev = self.ticket_proportion * tickets_sold * self.ticket_price
        merch_rev = tickets_sold * 0.20 * 35  # 20% of crowd buys $35 merch
        return self.flat_payment + ticket_rev + merch_rev

    def __repr__(self):
        return (f"Venue({self.name}, Ticket: ${self.ticket_price}, Cap: {self.capacity}, "
                f"Flat: ${self.flat_payment}, Share: {self.ticket_proportion:.2f})")


class City:
    """
    Represents a city with associated cost information and venues.
    """
    def __init__(self, name, avg_gas_price, toll_costs, parking_costs,
                 avg_meal_cost, avg_hotel_cost, regional_fanbase_strength):
        """
        Initialize a city.

        Parameters:
            avg_gas_price (float): Average gas price in the city.
            toll_costs (float): Toll costs within or to reach the city.
            parking_costs (float): Parking costs in the city.
            avg_meal_cost (float): Average meal cost per band member.
            avg_hotel_cost (float): Average hotel cost per room.
            regional_fanbase_strength (float): Band popularity in this city (0.0-1.0), used to scale expected pull in venues
        """
        self.name = name
        self.avg_gas_price = avg_gas_price
        self.toll_costs = toll_costs * NUM_VEHICLES
        self.parking_costs = parking_costs * NUM_VEHICLES
        self.avg_meal_cost = avg_meal_cost * NUM_BAND_MEMBERS
        self.avg_hotel_cost = avg_hotel_cost * ROOMS_NEEDED
        self.regional_fanbase_strength = regional_fanbase_strength  # Band popularity in this city
        self.venues = []  # List to hold Venue objects

    def add_venue(self, venue):
        """Add a venue to this city."""
        self.venues.append(venue)

    def total_expected_revenue(self):
        """Sum expected revenue from all venues in this city."""
        return sum(venue.expected_revenue(self.regional_fanbase_strength) for venue in self.venues)

    def __repr__(self):
        return f"City({self.name})"


# ============================================================================
# DISTANCE CALCULATION UTILITIES
# ============================================================================
def haversine_distance(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in miles.
    
    Parameters:
        coord1 (tuple): (longitude, latitude) of first point
        coord2 (tuple): (longitude, latitude) of second point
    
    Returns:
        float: Distance in miles
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    
    return c * r


# ============================================================================
# DATA LOADER
# ============================================================================


def create_cities_and_venues():
    """
    Create and initialize US national city and venue data with real 2025-2026 costs.
    
    Returns:
        tuple: (cities list, city_data dictionary, city_names list)
    """
    # --- Define cities with real 2025-2026 costs ---
    
    new_york = City("New York City", avg_gas_price=3.08, toll_costs=16, parking_costs=45,
                    avg_meal_cost=32, avg_hotel_cost=280, regional_fanbase_strength=0.48)
    los_angeles = City("Los Angeles", avg_gas_price=4.50, toll_costs=5, parking_costs=22,
                       avg_meal_cost=28, avg_hotel_cost=240, regional_fanbase_strength=0.78)
    nashville = City("Nashville", avg_gas_price=2.52, toll_costs=3, parking_costs=12,
                     avg_meal_cost=22, avg_hotel_cost=165, regional_fanbase_strength=0.28)
    austin = City("Austin", avg_gas_price=2.51, toll_costs=2, parking_costs=15,
                  avg_meal_cost=24, avg_hotel_cost=175, regional_fanbase_strength=0.90)
    chicago = City("Chicago", avg_gas_price=2.95, toll_costs=12, parking_costs=28,
                   avg_meal_cost=26, avg_hotel_cost=200, regional_fanbase_strength=0.34)
    new_orleans = City("New Orleans", avg_gas_price=2.52, toll_costs=3, parking_costs=12,
                       avg_meal_cost=23, avg_hotel_cost=160, regional_fanbase_strength=0.26)
    seattle = City("Seattle", avg_gas_price=3.96, toll_costs=8, parking_costs=22,
                   avg_meal_cost=28, avg_hotel_cost=205, regional_fanbase_strength=0.62)
    portland = City("Portland", avg_gas_price=3.57, toll_costs=4, parking_costs=18,
                    avg_meal_cost=26, avg_hotel_cost=180, regional_fanbase_strength=1)
    atlanta = City("Atlanta", avg_gas_price=2.65, toll_costs=4, parking_costs=15,
                   avg_meal_cost=21, avg_hotel_cost=150, regional_fanbase_strength=0.16)
    miami = City("Miami", avg_gas_price=2.95, toll_costs=7, parking_costs=20,
                 avg_meal_cost=29, avg_hotel_cost=220, regional_fanbase_strength=0.09)
    boston = City("Boston", avg_gas_price=3.00, toll_costs=12, parking_costs=35,
                  avg_meal_cost=30, avg_hotel_cost=270, regional_fanbase_strength=0.38)
    philadelphia = City("Philadelphia", avg_gas_price=3.12, toll_costs=8, parking_costs=22,
                        avg_meal_cost=26, avg_hotel_cost=185, regional_fanbase_strength=0.36)
    detroit = City("Detroit", avg_gas_price=2.85, toll_costs=6, parking_costs=12,
                   avg_meal_cost=22, avg_hotel_cost=145, regional_fanbase_strength=0.26)
    minneapolis = City("Minneapolis", avg_gas_price=2.75, toll_costs=4, parking_costs=18,
                       avg_meal_cost=24, avg_hotel_cost=170, regional_fanbase_strength=0.29)
    san_francisco = City("San Francisco", avg_gas_price=4.55, toll_costs=10, parking_costs=32,
                         avg_meal_cost=32, avg_hotel_cost=260, regional_fanbase_strength=0.5)
    denver = City("Denver", avg_gas_price=2.49, toll_costs=3, parking_costs=16,
                  avg_meal_cost=25, avg_hotel_cost=185, regional_fanbase_strength=0.41)
    memphis = City("Memphis", avg_gas_price=2.52, toll_costs=2, parking_costs=10,
                   avg_meal_cost=20, avg_hotel_cost=140, regional_fanbase_strength=0.0)
    oakland = City("Oakland", avg_gas_price=4.50, toll_costs=9, parking_costs=25,
                   avg_meal_cost=28, avg_hotel_cost=190, regional_fanbase_strength=0.5)
    cleveland = City("Cleveland", avg_gas_price=2.70, toll_costs=5, parking_costs=14,
                     avg_meal_cost=23, avg_hotel_cost=155, regional_fanbase_strength=0.28)
    washington = City("Washington", avg_gas_price=3.17, toll_costs=5, parking_costs=28,
                      avg_meal_cost=28, avg_hotel_cost=235, regional_fanbase_strength=0.22)
    st_louis = City("St. Louis", avg_gas_price=2.60, toll_costs=3, parking_costs=12,
                    avg_meal_cost=22, avg_hotel_cost=160, regional_fanbase_strength=0.28)
    birmingham = City("Birmingham", avg_gas_price=2.65, toll_costs=2, parking_costs=12,
                      avg_meal_cost=21, avg_hotel_cost=145, regional_fanbase_strength=0.32)
    knoxville = City("Knoxville", avg_gas_price=2.55, toll_costs=2, parking_costs=10,
                     avg_meal_cost=20, avg_hotel_cost=130, regional_fanbase_strength=0.44)
    chattanooga = City("Chattanooga", avg_gas_price=2.52, toll_costs=2, parking_costs=10,
                       avg_meal_cost=19, avg_hotel_cost=125, regional_fanbase_strength=0.56)
    louisville = City("Louisville", avg_gas_price=2.60, toll_costs=3, parking_costs=12,
                      avg_meal_cost=21, avg_hotel_cost=140, regional_fanbase_strength=0.76)
    huntsville = City("Huntsville", avg_gas_price=2.50, toll_costs=2, parking_costs=8,
                      avg_meal_cost=19, avg_hotel_cost=120, regional_fanbase_strength=0.28)
    jackson = City("Jackson", avg_gas_price=2.45, toll_costs=1, parking_costs=8,
                   avg_meal_cost=18, avg_hotel_cost=115, regional_fanbase_strength=0.0)
    fayetteville = City("Fayetteville", avg_gas_price=2.48, toll_costs=0, parking_costs=10,
                        avg_meal_cost=18, avg_hotel_cost=130, regional_fanbase_strength=1)


    if INTERNATIONAL: 
        london = City("London", avg_gas_price=7.20, toll_costs=18, parking_costs=35,
                  avg_meal_cost=30, avg_hotel_cost=247, regional_fanbase_strength=0.55)
        manchester = City("Manchester", avg_gas_price=7.15, toll_costs=8, parking_costs=18,
                        avg_meal_cost=24, avg_hotel_cost=160, regional_fanbase_strength=0.48)
        berlin = City("Berlin", avg_gas_price=6.85, toll_costs=5, parking_costs=15,
                    avg_meal_cost=22, avg_hotel_cost=145, regional_fanbase_strength=0.72)
        paris = City("Paris", avg_gas_price=7.10, toll_costs=12, parking_costs=28,
                    avg_meal_cost=28, avg_hotel_cost=174, regional_fanbase_strength=0.62)
        amsterdam = City("Amsterdam", avg_gas_price=7.35, toll_costs=6, parking_costs=32,
                        avg_meal_cost=26, avg_hotel_cost=140, regional_fanbase_strength=0.58)
        dublin = City("Dublin", avg_gas_price=6.95, toll_costs=8, parking_costs=22,
                    avg_meal_cost=27, avg_hotel_cost=165, regional_fanbase_strength=0.51)
        hamburg = City("Hamburg", avg_gas_price=6.88, toll_costs=4, parking_costs=16,
                    avg_meal_cost=23, avg_hotel_cost=150, regional_fanbase_strength=0.44)
        barcelona = City("Barcelona", avg_gas_price=6.50, toll_costs=8, parking_costs=20,
                        avg_meal_cost=24, avg_hotel_cost=130, regional_fanbase_strength=0.54)
        brussels = City("Brussels", avg_gas_price=6.75, toll_costs=5, parking_costs=18,
                        avg_meal_cost=25, avg_hotel_cost=135, regional_fanbase_strength=0.42)
        vienna = City("Vienna", avg_gas_price=6.60, toll_costs=6, parking_costs=16,
                    avg_meal_cost=24, avg_hotel_cost=140, regional_fanbase_strength=0.47)
        glasgow = City("Glasgow", avg_gas_price=7.18, toll_costs=6, parking_costs=15,
                    avg_meal_cost=22, avg_hotel_cost=145, regional_fanbase_strength=0.39)
        copenhagen = City("Copenhagen", avg_gas_price=7.85, toll_costs=8, parking_costs=30,
                        avg_meal_cost=32, avg_hotel_cost=190, regional_fanbase_strength=0.45)
        prague = City("Prague", avg_gas_price=6.20, toll_costs=4, parking_costs=12,
                    avg_meal_cost=18, avg_hotel_cost=110, regional_fanbase_strength=0.50)
        budapest = City("Budapest", avg_gas_price=6.10, toll_costs=3, parking_costs=10,
                        avg_meal_cost=16, avg_hotel_cost=95, regional_fanbase_strength=0.46)
        stockholm = City("Stockholm", avg_gas_price=7.75, toll_costs=10, parking_costs=28,
                     avg_meal_cost=30, avg_hotel_cost=175, regional_fanbase_strength=0.43)

    
    
    
        
    cities = [new_york, los_angeles, nashville, austin, chicago, new_orleans,
              seattle, portland, atlanta, miami, boston, philadelphia,
              detroit, minneapolis, san_francisco, denver, memphis,
              oakland, cleveland, washington, st_louis, birmingham, knoxville,
              chattanooga, louisville, huntsville, jackson, fayetteville]
    
    if INTERNATIONAL:
        cities = [new_york, los_angeles, nashville, austin, chicago, new_orleans,
              seattle, portland, atlanta, miami, boston, philadelphia,
              detroit, minneapolis, san_francisco, denver, memphis,
              oakland, cleveland, washington, st_louis, birmingham, knoxville,
              chattanooga, louisville, huntsville, jackson, fayetteville, london, manchester, berlin, paris, amsterdam, dublin, hamburg,
              barcelona, brussels, vienna, glasgow, copenhagen, prague, 
              budapest, stockholm,]
    
        
        
    city_data = {city.name: city for city in cities}
    city_names = [city.name for city in cities]

    # --- Add venues for each city ---
    # New York City
    new_york.add_venue(Venue("Madison Square Garden", ticket_price=75, capacity=20000,
                              flat_payment=0, ticket_proportion=0.30, popularity_threshold=0.85))
    new_york.add_venue(Venue("Mercury Lounge", ticket_price=15, capacity=250,
                              flat_payment=200, ticket_proportion=0.50, popularity_threshold=0.05))
    new_york.add_venue(Venue("Bowery Ballroom", ticket_price=25, capacity=575,
                              flat_payment=500, ticket_proportion=0.35, popularity_threshold=0.15))
    new_york.add_venue(Venue("Radio City Music Hall", ticket_price=50, capacity=6000,
                              flat_payment=1000, ticket_proportion=0.28, popularity_threshold=0.45))

    # Los Angeles
    los_angeles.add_venue(Venue("Hollywood Bowl", ticket_price=60, capacity=17500,
                                 flat_payment=10000, ticket_proportion=0.25, popularity_threshold=0.80))
    los_angeles.add_venue(Venue("The Smell", ticket_price=10, capacity=150,
                                 flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.05))
    los_angeles.add_venue(Venue("Teragram Ballroom", ticket_price=20, capacity=600,
                                 flat_payment=300, ticket_proportion=0.30, popularity_threshold=0.12))
    los_angeles.add_venue(Venue("Greek Theatre", ticket_price=40, capacity=5900,
                                 flat_payment=1500, ticket_proportion=0.25, popularity_threshold=0.40))

    # Nashville
    nashville.add_venue(Venue("Ryman Auditorium", ticket_price=45, capacity=5000,
                               flat_payment=0, ticket_proportion=0.30, popularity_threshold=0.70))
    nashville.add_venue(Venue("The End", ticket_price=12, capacity=150,
                               flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.05))
    nashville.add_venue(Venue("Exit/In", ticket_price=18, capacity=500,
                               flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.10))
    nashville.add_venue(Venue("Ascend Amphitheater", ticket_price=40, capacity=6800,
                               flat_payment=2500, ticket_proportion=0.20, popularity_threshold=0.45))

    # Austin
    austin.add_venue(Venue("Moody Center", ticket_price=50, capacity=15000,
                            flat_payment=5000, ticket_proportion=0.20, popularity_threshold=0.75))
    austin.add_venue(Venue("Hole in the Wall", ticket_price=10, capacity=120,
                            flat_payment=75, ticket_proportion=0.60, popularity_threshold=0.04))
    austin.add_venue(Venue("Mohawk Austin", ticket_price=18, capacity=700,
                            flat_payment=250, ticket_proportion=0.30, popularity_threshold=0.10))
    austin.add_venue(Venue("ACL Live at Moody Theater", ticket_price=35, capacity=2750,
                            flat_payment=1000, ticket_proportion=0.25, popularity_threshold=0.38))

    # Chicago
    chicago.add_venue(Venue("United Center", ticket_price=55, capacity=23000,
                            flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.85))
    chicago.add_venue(Venue("The Hideout", ticket_price=12, capacity=150,
                            flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.06))
    chicago.add_venue(Venue("Lincoln Hall", ticket_price=20, capacity=500,
                            flat_payment=300, ticket_proportion=0.40, popularity_threshold=0.13))
    chicago.add_venue(Venue("Aragon Ballroom", ticket_price=35, capacity=5000,
                            flat_payment=2000, ticket_proportion=0.25, popularity_threshold=0.35))

    # New Orleans
    new_orleans.add_venue(Venue("Smoothie King Center", ticket_price=45, capacity=16000,
                                 flat_payment=2000, ticket_proportion=0.22, popularity_threshold=0.75))
    new_orleans.add_venue(Venue("Siberia", ticket_price=10, capacity=120,
                                 flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.05))
    new_orleans.add_venue(Venue("Tipitina's", ticket_price=18, capacity=600,
                                 flat_payment=250, ticket_proportion=0.35, popularity_threshold=0.12))
    new_orleans.add_venue(Venue("The Joy Theater", ticket_price=30, capacity=1200,
                                 flat_payment=800, ticket_proportion=0.28, popularity_threshold=0.30))

    # Seattle
    seattle.add_venue(Venue("Climate Pledge Arena", ticket_price=55, capacity=18000,
                              flat_payment=0, ticket_proportion=0.28, popularity_threshold=0.80))
    seattle.add_venue(Venue("The Sunset", ticket_price=12, capacity=200,
                              flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.06))
    seattle.add_venue(Venue("Neumos", ticket_price=20, capacity=750,
                              flat_payment=300, ticket_proportion=0.30, popularity_threshold=0.14))
    seattle.add_venue(Venue("The Moore Theatre", ticket_price=32, capacity=1400,
                              flat_payment=900, ticket_proportion=0.25, popularity_threshold=0.32))

    # Portland
    portland.add_venue(Venue("Moda Center", ticket_price=50, capacity=19500,
                              flat_payment=5000, ticket_proportion=0.24, popularity_threshold=0.78))
    portland.add_venue(Venue("The Know", ticket_price=10, capacity=100,
                              flat_payment=80, ticket_proportion=0.50, popularity_threshold=0.05))
    portland.add_venue(Venue("Doug Fir Lounge", ticket_price=18, capacity=500,
                              flat_payment=250, ticket_proportion=0.35, popularity_threshold=0.11))
    portland.add_venue(Venue("Crystal Ballroom", ticket_price=30, capacity=1500,
                              flat_payment=800, ticket_proportion=0.25, popularity_threshold=0.28))

    # Atlanta
    atlanta.add_venue(Venue("State Farm Arena", ticket_price=50, capacity=21000,
                              flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.80))
    atlanta.add_venue(Venue("529 Bar", ticket_price=10, capacity=100,
                              flat_payment=100, ticket_proportion=0.60, popularity_threshold=0.04))
    atlanta.add_venue(Venue("The Earl", ticket_price=18, capacity=300,
                              flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.10))
    atlanta.add_venue(Venue("Variety Playhouse", ticket_price=30, capacity=1100,
                              flat_payment=600, ticket_proportion=0.28, popularity_threshold=0.30))

    # Miami
    miami.add_venue(Venue("Kaseya Center", ticket_price=50, capacity=19600,
                            flat_payment=3000, ticket_proportion=0.20, popularity_threshold=0.78))
    miami.add_venue(Venue("Gramps", ticket_price=12, capacity=200,
                            flat_payment=150, ticket_proportion=0.45, popularity_threshold=0.06))
    miami.add_venue(Venue("The Ground", ticket_price=20, capacity=600,
                            flat_payment=250, ticket_proportion=0.32, popularity_threshold=0.14))
    miami.add_venue(Venue("The Fillmore Miami", ticket_price=35, capacity=2700,
                            flat_payment=1200, ticket_proportion=0.26, popularity_threshold=0.35))

    # Boston
    boston.add_venue(Venue("TD Garden", ticket_price=60, capacity=19000,
                             flat_payment=0, ticket_proportion=0.27, popularity_threshold=0.82))
    boston.add_venue(Venue("Great Scott", ticket_price=12, capacity=150,
                             flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.05))
    boston.add_venue(Venue("Paradise Rock Club", ticket_price=22, capacity=725,
                             flat_payment=300, ticket_proportion=0.35, popularity_threshold=0.13))
    boston.add_venue(Venue("House of Blues", ticket_price=38, capacity=2400,
                             flat_payment=1500, ticket_proportion=0.25, popularity_threshold=0.40))

    # Philadelphia
    philadelphia.add_venue(Venue("Wells Fargo Center", ticket_price=50, capacity=21000,
                                  flat_payment=4000, ticket_proportion=0.23, popularity_threshold=0.78))
    philadelphia.add_venue(Venue("Johnny Brenda's", ticket_price=15, capacity=200,
                                  flat_payment=150, ticket_proportion=0.45, popularity_threshold=0.06))
    philadelphia.add_venue(Venue("Union Transfer", ticket_price=25, capacity=1200,
                                  flat_payment=400, ticket_proportion=0.30, popularity_threshold=0.16))
    philadelphia.add_venue(Venue("The Fillmore Philly", ticket_price=42, capacity=2600,
                                  flat_payment=1200, ticket_proportion=0.22, popularity_threshold=0.38))

    # Detroit
    detroit.add_venue(Venue("Little Caesars Arena", ticket_price=45, capacity=20000,
                               flat_payment=0, ticket_proportion=0.22, popularity_threshold=0.75))
    detroit.add_venue(Venue("Psycho Suzi's Motor Lounge", ticket_price=10, capacity=150,
                               flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.05))
    detroit.add_venue(Venue("El Club", ticket_price=15, capacity=300,
                               flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.10))
    detroit.add_venue(Venue("St. Andrew's Hall", ticket_price=30, capacity=1200,
                               flat_payment=800, ticket_proportion=0.28, popularity_threshold=0.32))

    # Minneapolis
    minneapolis.add_venue(Venue("Target Center", ticket_price=50, capacity=20000,
                                  flat_payment=2000, ticket_proportion=0.24, popularity_threshold=0.80))
    minneapolis.add_venue(Venue("7th St Entry", ticket_price=12, capacity=250,
                                  flat_payment=150, ticket_proportion=0.50, popularity_threshold=0.05))
    minneapolis.add_venue(Venue("First Avenue", ticket_price=22, capacity=1500,
                                  flat_payment=500, ticket_proportion=0.35, popularity_threshold=0.14))
    minneapolis.add_venue(Venue("Fillmore Minneapolis", ticket_price=38, capacity=2600,
                                  flat_payment=1500, ticket_proportion=0.22, popularity_threshold=0.38))

    # San Francisco
    san_francisco.add_venue(Venue("Chase Center", ticket_price=65, capacity=18000,
                                 flat_payment=0, ticket_proportion=0.30, popularity_threshold=0.85))
    san_francisco.add_venue(Venue("Slim's", ticket_price=18, capacity=500,
                                 flat_payment=300, ticket_proportion=0.35, popularity_threshold=0.12))
    san_francisco.add_venue(Venue("Bill Graham Civic Auditorium", ticket_price=55, capacity=8500,
                                 flat_payment=1000, ticket_proportion=0.30, popularity_threshold=0.70))
    san_francisco.add_venue(Venue("The Warfield", ticket_price=40, capacity=2300,
                                 flat_payment=800, ticket_proportion=0.28, popularity_threshold=0.35))

    # Denver
    denver.add_venue(Venue("Ball Arena", ticket_price=50, capacity=20000,
                                 flat_payment=0, ticket_proportion=0.26, popularity_threshold=0.85))
    denver.add_venue(Venue("Bluebird Theater", ticket_price=18, capacity=550,
                                 flat_payment=300, ticket_proportion=0.35, popularity_threshold=0.12))
    denver.add_venue(Venue("Mission Ballroom", ticket_price=32, capacity=3000,
                                 flat_payment=1000, ticket_proportion=0.28, popularity_threshold=0.36))
    denver.add_venue(Venue("Red Rocks", ticket_price=50, capacity=9500,
                                 flat_payment=1000, ticket_proportion=0.30, popularity_threshold=0.84))

    # Memphis
    memphis.add_venue(Venue("Orpheum Theatre", ticket_price=35, capacity=2400,
                                 flat_payment=1000, ticket_proportion=0.35, popularity_threshold=0.70))
    memphis.add_venue(Venue("Hi-Tone Cafe", ticket_price=15, capacity=300,
                                 flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.10))
    memphis.add_venue(Venue("FedExForum", ticket_price=60, capacity=19000,
                                 flat_payment=1200, ticket_proportion=0.30, popularity_threshold=0.80))
    memphis.add_venue(Venue("Levitt Shell", ticket_price=30, capacity=2500,
                                 flat_payment=700, ticket_proportion=0.28, popularity_threshold=0.62))

    # Oakland
    oakland.add_venue(Venue("Oracle Arena", ticket_price=45, capacity=19500,
                                 flat_payment=0, ticket_proportion=0.22, popularity_threshold=0.75))
    oakland.add_venue(Venue("Starline Social Club", ticket_price=15, capacity=400,
                                 flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.10))
    oakland.add_venue(Venue("Fox Theater Oakland", ticket_price=30, capacity=2800,
                                 flat_payment=1200, ticket_proportion=0.25, popularity_threshold=0.38))
    oakland.add_venue(Venue("Oakland Arena", ticket_price=55, capacity=19000,
                                 flat_payment=1000, ticket_proportion=0.30, popularity_threshold=0.60))

    # Cleveland
    cleveland.add_venue(Venue("Rocket Mortgage FieldHouse", ticket_price=50, capacity=19700,
                                 flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.80))
    cleveland.add_venue(Venue("Mahall's", ticket_price=15, capacity=350,
                                 flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.12))
    cleveland.add_venue(Venue("Beachland Ballroom", ticket_price=28, capacity=1200,
                                 flat_payment=800, ticket_proportion=0.28, popularity_threshold=0.30))
    cleveland.add_venue(Venue("Jacobs Pavilion at Nautica", ticket_price=45, capacity=5000,
                                 flat_payment=1200, ticket_proportion=0.30, popularity_threshold=0.48))

    # Washington, D.C.
    washington.add_venue(Venue("Capital One Arena", ticket_price=55, capacity=20000,
                                 flat_payment=0, ticket_proportion=0.27, popularity_threshold=0.82))
    washington.add_venue(Venue("Black Cat", ticket_price=12, capacity=200,
                                 flat_payment=150, ticket_proportion=0.50, popularity_threshold=0.05))
    washington.add_venue(Venue("9:30 Club", ticket_price=22, capacity=1200,
                                 flat_payment=500, ticket_proportion=0.35, popularity_threshold=0.20))
    washington.add_venue(Venue("The Anthem", ticket_price=38, capacity=6500,
                                 flat_payment=1500, ticket_proportion=0.25, popularity_threshold=0.40))

    # St. Louis
    st_louis.add_venue(Venue("Enterprise Center", ticket_price=48, capacity=19000,
                                 flat_payment=0, ticket_proportion=0.24, popularity_threshold=0.78))
    st_louis.add_venue(Venue("Delmar Hall", ticket_price=18, capacity=800,
                                 flat_payment=300, ticket_proportion=0.35, popularity_threshold=0.12))
    st_louis.add_venue(Venue("Hollywood Casino Ampitheatre", ticket_price=50, capacity=20000,
                                 flat_payment=1000, ticket_proportion=0.26, popularity_threshold=0.75))
    st_louis.add_venue(Venue("Stifel Theatre", ticket_price=40, capacity=3100,
                                 flat_payment=900, ticket_proportion=0.25, popularity_threshold=0.30))
    
    
    birmingham.add_venue(Venue("Legacy Arena", ticket_price=45, capacity=4000,
                               flat_payment=2000, ticket_proportion=0.20, popularity_threshold=0.65))
    birmingham.add_venue(Venue("Saturn Birmingham", ticket_price=18, capacity=500,
                               flat_payment=300, ticket_proportion=0.25, popularity_threshold=0.18))
    birmingham.add_venue(Venue("Iron City Bham", ticket_price=32, capacity=1300,
                               flat_payment=700, ticket_proportion=0.20, popularity_threshold=0.40))
    knoxville.add_venue(Venue("Knox Coliseum", ticket_price=42, capacity=3500,
                              flat_payment=1000, ticket_proportion=0.15, popularity_threshold=0.50))
    knoxville.add_venue(Venue("The Mill & Mine", ticket_price=22, capacity=1200,
                              flat_payment=400, ticket_proportion=0.25, popularity_threshold=0.28))
    knoxville.add_venue(Venue("Open Chord Stage", ticket_price=15, capacity=300,
                              flat_payment=200, ticket_proportion=0.20, popularity_threshold=0.12))

    chattanooga.add_venue(Venue("MacArthur Park Amphitheater", ticket_price=38, capacity=3000,
                                 flat_payment=0, ticket_proportion=0.30, popularity_threshold=0.55))
    chattanooga.add_venue(Venue("Songbirds", ticket_price=18, capacity=400,
                                 flat_payment=150, ticket_proportion=0.25, popularity_threshold=0.15))
    chattanooga.add_venue(Venue("Walker Theatre", ticket_price=30, capacity=900,
                                 flat_payment=500, ticket_proportion=0.20, popularity_threshold=0.32))

    louisville.add_venue(Venue("KFC Yum! Center", ticket_price=48, capacity=5000,
                               flat_payment=3000, ticket_proportion=0.10, popularity_threshold=0.70))
    louisville.add_venue(Venue("Headliners Music Hall", ticket_price=25, capacity=600,
                               flat_payment=300, ticket_proportion=0.20, popularity_threshold=0.20))
    louisville.add_venue(Venue("Zanzabar", ticket_price=15, capacity=300,
                               flat_payment=150, ticket_proportion=0.25, popularity_threshold=0.10))

    huntsville.add_venue(Venue("Von Braun Center", ticket_price=45, capacity=4500,
                               flat_payment=500, ticket_proportion=0.20, popularity_threshold=0.65))
    huntsville.add_venue(Venue("Mars Music Hall", ticket_price=28, capacity=1500,
                               flat_payment=700, ticket_proportion=0.20, popularity_threshold=0.30))
    huntsville.add_venue(Venue("The Orion Amphitheater", ticket_price=42, capacity=8000,
                               flat_payment=2000, ticket_proportion=0.15, popularity_threshold=0.50))

    jackson.add_venue(Venue("Jackson Convention Center", ticket_price=40, capacity=4000,
                            flat_payment=0, ticket_proportion=0.30, popularity_threshold=0.55))
    jackson.add_venue(Venue("Duling Hall", ticket_price=18, capacity=350,
                            flat_payment=150, ticket_proportion=0.25, popularity_threshold=0.12))
    jackson.add_venue(Venue("Hal & Mal's", ticket_price=15, capacity=300,
                            flat_payment=100, ticket_proportion=0.20, popularity_threshold=0.08))

    fayetteville.add_venue(Venue("George's Majestic Lounge", ticket_price=18, capacity=700,
                                  flat_payment=300, ticket_proportion=0.25, popularity_threshold=0.10))
    fayetteville.add_venue(Venue("JJ's Live", ticket_price=18, capacity=1000,
                                  flat_payment=1000, ticket_proportion=0.20, popularity_threshold=0.30))
    fayetteville.add_venue(Venue("Nomad's Trailside", ticket_price=8, capacity=200,
                                  flat_payment=0, ticket_proportion=1, popularity_threshold=0))
    
    # Add venues (4 per city with realistic European ticket prices)
    if INTERNATIONAL: 
        
        # London
        london.add_venue(Venue("O2 Arena", ticket_price=65, capacity=20000,
                            flat_payment=0, ticket_proportion=0.28, popularity_threshold=0.85))
        london.add_venue(Venue("Brixton Academy", ticket_price=28, capacity=4900,
                            flat_payment=2000, ticket_proportion=0.30, popularity_threshold=0.40))
        london.add_venue(Venue("KOKO", ticket_price=18, capacity=1500,
                            flat_payment=800, ticket_proportion=0.35, popularity_threshold=0.18))
        london.add_venue(Venue("The Lexington", ticket_price=12, capacity=200,
                            flat_payment=150, ticket_proportion=0.50, popularity_threshold=0.08))
        
        # Manchester
        manchester.add_venue(Venue("AO Arena", ticket_price=50, capacity=21000,
                                flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.82))
        manchester.add_venue(Venue("O2 Apollo Manchester", ticket_price=25, capacity=3500,
                                flat_payment=1500, ticket_proportion=0.28, popularity_threshold=0.35))
        manchester.add_venue(Venue("Albert Hall", ticket_price=18, capacity=2000,
                                flat_payment=700, ticket_proportion=0.32, popularity_threshold=0.16))
        manchester.add_venue(Venue("Night and Day Cafe", ticket_price=10, capacity=150,
                                flat_payment=100, ticket_proportion=0.50, popularity_threshold=0.06))
        
        # Berlin
        berlin.add_venue(Venue("Mercedes-Benz Arena", ticket_price=48, capacity=17000,
                            flat_payment=0, ticket_proportion=0.26, popularity_threshold=0.78))
        berlin.add_venue(Venue("Astra Kulturhaus", ticket_price=20, capacity=1600,
                            flat_payment=600, ticket_proportion=0.30, popularity_threshold=0.25))
        berlin.add_venue(Venue("SO36", ticket_price=15, capacity=800,
                            flat_payment=300, ticket_proportion=0.35, popularity_threshold=0.12))
        berlin.add_venue(Venue("Madame Claude", ticket_price=8, capacity=150,
                            flat_payment=80, ticket_proportion=0.50, popularity_threshold=0.05))
        
        # Paris
        paris.add_venue(Venue("AccorHotels Arena", ticket_price=58, capacity=20300,
                            flat_payment=0, ticket_proportion=0.27, popularity_threshold=0.82))
        paris.add_venue(Venue("Olympia", ticket_price=32, capacity=2000,
                            flat_payment=1200, ticket_proportion=0.28, popularity_threshold=0.38))
        paris.add_venue(Venue("La Cigale", ticket_price=22, capacity=1400,
                            flat_payment=700, ticket_proportion=0.32, popularity_threshold=0.20))
        paris.add_venue(Venue("Le Pop-Up du Label", ticket_price=12, capacity=200,
                            flat_payment=120, ticket_proportion=0.45, popularity_threshold=0.07))
        
        # Amsterdam
        amsterdam.add_venue(Venue("Ziggo Dome", ticket_price=52, capacity=17000,
                                flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.80))
        amsterdam.add_venue(Venue("AFAS Live", ticket_price=28, capacity=6000,
                                flat_payment=1800, ticket_proportion=0.27, popularity_threshold=0.42))
        amsterdam.add_venue(Venue("Paradiso", ticket_price=18, capacity=1500,
                                flat_payment=600, ticket_proportion=0.33, popularity_threshold=0.18))
        amsterdam.add_venue(Venue("Cafe de Ceuvel", ticket_price=10, capacity=180,
                                flat_payment=100, ticket_proportion=0.48, popularity_threshold=0.06))
        
        # Dublin
        dublin.add_venue(Venue("3Arena", ticket_price=45, capacity=13000,
                            flat_payment=0, ticket_proportion=0.26, popularity_threshold=0.75))
        dublin.add_venue(Venue("Vicar Street", ticket_price=25, capacity=1050,
                            flat_payment=500, ticket_proportion=0.30, popularity_threshold=0.28))
        dublin.add_venue(Venue("Whelan's", ticket_price=15, capacity=450,
                            flat_payment=250, ticket_proportion=0.35, popularity_threshold=0.12))
        dublin.add_venue(Venue("The Grand Social", ticket_price=10, capacity=250,
                            flat_payment=120, ticket_proportion=0.45, popularity_threshold=0.07))
        
        # Hamburg
        hamburg.add_venue(Venue("Barclays Arena", ticket_price=45, capacity=16000,
                                flat_payment=0, ticket_proportion=0.24, popularity_threshold=0.78))
        hamburg.add_venue(Venue("Grosse Freiheit 36", ticket_price=20, capacity=1300,
                                flat_payment=550, ticket_proportion=0.30, popularity_threshold=0.22))
        hamburg.add_venue(Venue("Molotow", ticket_price=12, capacity=400,
                                flat_payment=200, ticket_proportion=0.35, popularity_threshold=0.10))
        hamburg.add_venue(Venue("Pooca Bar", ticket_price=8, capacity=100,
                                flat_payment=75, ticket_proportion=0.50, popularity_threshold=0.05))
        
        # Barcelona
        barcelona.add_venue(Venue("Palau Sant Jordi", ticket_price=42, capacity=17000,
                                flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.75))
        barcelona.add_venue(Venue("Razzmatazz", ticket_price=20, capacity=3000,
                                flat_payment=900, ticket_proportion=0.28, popularity_threshold=0.32))
        barcelona.add_venue(Venue("Sala Apolo", ticket_price=15, capacity=1500,
                                flat_payment=500, ticket_proportion=0.32, popularity_threshold=0.15))
        barcelona.add_venue(Venue("Heliogabal", ticket_price=8, capacity=80,
                                flat_payment=60, ticket_proportion=0.50, popularity_threshold=0.04))
        
        # Brussels
        brussels.add_venue(Venue("Forest National", ticket_price=38, capacity=8000,
                                flat_payment=0, ticket_proportion=0.24, popularity_threshold=0.72))
        brussels.add_venue(Venue("Ancienne Belgique", ticket_price=22, capacity=2000,
                                flat_payment=800, ticket_proportion=0.30, popularity_threshold=0.30))
        brussels.add_venue(Venue("Botanique", ticket_price=18, capacity=1200,
                                flat_payment=450, ticket_proportion=0.32, popularity_threshold=0.16))
        brussels.add_venue(Venue("Madame Moustache", ticket_price=10, capacity=200,
                                flat_payment=100, ticket_proportion=0.45, popularity_threshold=0.06))
        
        # Vienna
        vienna.add_venue(Venue("Wiener Stadthalle", ticket_price=42, capacity=16000,
                            flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.76))
        vienna.add_venue(Venue("Arena Wien", ticket_price=22, capacity=3000,
                            flat_payment=1000, ticket_proportion=0.28, popularity_threshold=0.34))
        vienna.add_venue(Venue("Flex", ticket_price=15, capacity=750,
                            flat_payment=350, ticket_proportion=0.33, popularity_threshold=0.14))
        vienna.add_venue(Venue("Chelsea", ticket_price=10, capacity=300,
                            flat_payment=120, ticket_proportion=0.45, popularity_threshold=0.07))
        
        # Glasgow
        glasgow.add_venue(Venue("OVO Hydro", ticket_price=45, capacity=14300,
                                flat_payment=0, ticket_proportion=0.24, popularity_threshold=0.78))
        glasgow.add_venue(Venue("Barrowland Ballroom", ticket_price=20, capacity=1900,
                                flat_payment=650, ticket_proportion=0.30, popularity_threshold=0.28))
        glasgow.add_venue(Venue("King Tut's Wah Wah Hut", ticket_price=12, capacity=300,
                                flat_payment=180, ticket_proportion=0.35, popularity_threshold=0.12))
        glasgow.add_venue(Venue("Nice 'N' Sleazy", ticket_price=8, capacity=150,
                                flat_payment=80, ticket_proportion=0.48, popularity_threshold=0.05))
        
        # Copenhagen
        copenhagen.add_venue(Venue("Royal Arena", ticket_price=55, capacity=16000,
                                flat_payment=0, ticket_proportion=0.26, popularity_threshold=0.80))
        copenhagen.add_venue(Venue("Vega", ticket_price=28, capacity=1500,
                                flat_payment=750, ticket_proportion=0.30, popularity_threshold=0.32))
        copenhagen.add_venue(Venue("Loppen", ticket_price=15, capacity=450,
                                flat_payment=220, ticket_proportion=0.35, popularity_threshold=0.12))
        copenhagen.add_venue(Venue("Stengade", ticket_price=10, capacity=200,
                                flat_payment=100, ticket_proportion=0.45, popularity_threshold=0.06))
        
        # Prague
        prague.add_venue(Venue("O2 Arena Prague", ticket_price=35, capacity=18000,
                            flat_payment=0, ticket_proportion=0.23, popularity_threshold=0.75))
        prague.add_venue(Venue("Forum Karlin", ticket_price=18, capacity=3000,
                            flat_payment=700, ticket_proportion=0.28, popularity_threshold=0.30))
        prague.add_venue(Venue("MeetFactory", ticket_price=12, capacity=800,
                            flat_payment=300, ticket_proportion=0.32, popularity_threshold=0.14))
        prague.add_venue(Venue("Cross Club", ticket_price=8, capacity=400,
                            flat_payment=150, ticket_proportion=0.40, popularity_threshold=0.08))
        
        # Budapest
        budapest.add_venue(Venue("Papp Laszlo Budapest Arena", ticket_price=32, capacity=12500,
                                flat_payment=0, ticket_proportion=0.22, popularity_threshold=0.72))
        budapest.add_venue(Venue("Barba Negra", ticket_price=18, capacity=2500,
                                flat_payment=600, ticket_proportion=0.28, popularity_threshold=0.28))
        budapest.add_venue(Venue("A38", ticket_price=12, capacity=800,
                                flat_payment=280, ticket_proportion=0.32, popularity_threshold=0.15))
        budapest.add_venue(Venue("Dürer Kert", ticket_price=8, capacity=500,
                                flat_payment=150, ticket_proportion=0.40, popularity_threshold=0.08))
        
        # Stockholm
        stockholm.add_venue(Venue("Avicii Arena", ticket_price=48, capacity=16000,
                                flat_payment=0, ticket_proportion=0.25, popularity_threshold=0.78))
        stockholm.add_venue(Venue("Annexet", ticket_price=25, capacity=5000,
                                flat_payment=1200, ticket_proportion=0.28, popularity_threshold=0.38))
        stockholm.add_venue(Venue("Debaser Strand", ticket_price=15, capacity=1200,
                                flat_payment=450, ticket_proportion=0.32, popularity_threshold=0.16))
        stockholm.add_venue(Venue("Nalen", ticket_price=10, capacity=800,
                                flat_payment=250, ticket_proportion=0.38, popularity_threshold=0.10))
        


    return cities, city_data, city_names


def create_city_coords():
    """international city coord"""
    return {
        "New York City": (-74.0060, 40.7128),
        "Los Angeles": (-118.2437, 34.0522),
        "Nashville": (-86.7816, 36.1627),
        "Austin": (-97.7431, 30.2672),
        "Chicago": (-87.6298, 41.8781),
        "New Orleans": (-90.0715, 29.9511),
        "Seattle": (-122.3321, 47.6062),
        "Portland": (-122.6765, 45.5231),
        "Atlanta": (-84.3880, 33.7490),
        "Miami": (-80.1918, 25.7617),
        "Boston": (-71.0589, 42.3601),
        "Philadelphia": (-75.1652, 39.9526),
        "Detroit": (-83.0458, 42.3314),
        "Minneapolis": (-93.2650, 44.9778),
        "San Francisco": (-122.4194, 37.7749),
        "Denver": (-104.9903, 39.7392),
        "Memphis": (-90.0490, 35.1495),
        "Oakland": (-122.2712, 37.8044),
        "Cleveland": (-81.6944, 41.4993),
        "Washington": (-77.0369, 38.9072),
        "St. Louis": (-90.1994, 38.6270), 
        "Birmingham": (-86.80249, 33.52066),
        "Knoxville": (-83.92074, 35.96064),
        "Chattanooga": (-85.30968, 35.04563),
        "Louisville": (-85.75846, 38.25267),
        "Huntsville": (-86.58610, 34.73037),
        "Jackson": (-90.18481, 32.29876),
        "Fayetteville": (-94.15743, 36.06258),
        "London": (-0.1276, 51.5074),
        "Manchester": (-2.2426, 53.4808),
        "Berlin": (13.4050, 52.5200),
        "Paris": (2.3522, 48.8566),
        "Amsterdam": (4.9041, 52.3676),
        "Dublin": (-6.2603, 53.3498),
        "Hamburg": (9.9937, 53.5511),
        "Barcelona": (2.1734, 41.3851),
        "Brussels": (4.3517, 50.8503),
        "Vienna": (16.3738, 48.2082),
        "Glasgow": (-4.2518, 55.8642),
        "Copenhagen": (12.5683, 55.6761),
        "Prague": (14.4378, 50.0755),
        "Budapest": (19.0402, 47.4979),
        "Stockholm": (18.0686, 59.3293),
    }




# ============================================================================
# DISTANCE MATRIX GENERATION
# ============================================================================
def create_distance_matrix(city_names):
    """
    Create distance matrix using scaled haversine distance.
    Automatically detects which dataset is being used based on city names.
    
    Parameters:
        city_names (list): List of city names
    
    Returns:
        dict: Distance matrix with (city1, city2) tuples as keys and distances as values
    """
    # Get appropriate coordinates
    city_coords = create_city_coords()
    
    distance_matrix = {}
    
    # Highway factor to convert straight-line to approximate driving distance
    HIGHWAY_FACTOR = 1.3
    
    for city1 in city_names:
        for city2 in city_names:
            if city1 == city2:
                distance_matrix[(city1, city2)] = 0
            else:
                coord1 = city_coords[city1]
                coord2 = city_coords[city2]
                straight_line_distance = haversine_distance(coord1, coord2)
                highway_distance = straight_line_distance * HIGHWAY_FACTOR
                distance_matrix[(city1, city2)] = round(highway_distance)
    
    return distance_matrix

# =============================================================================
# MODEL DEFINITION WITH FLIGHT TRANSPORTATION
# =============================================================================
def create_optimization_model(cities, city_data, distance_matrix):
    """
    Create optimization model with optional flight transportation.
    
    Flights automatically become available for distances exceeding 
    FLIGHT_DISTANCE_THRESHOLD (configured globally). The model chooses
    the most cost-effective transportation method.
    
    Parameters:
        cities (list): List of City objects
        city_data (dict): Dictionary mapping city names to City objects
        distance_matrix (dict): Distance matrix
    
    Returns:
        tuple: (model, toll_parking_total, refuel_total, staying_total, flight_total)
    """
    model = pyo.ConcreteModel()

    # Sets
    model.CITIES = pyo.Set(initialize=[c.name for c in cities])
    model.EDGES = pyo.Set(
        initialize=[(i, j) for i in model.CITIES for j in model.CITIES if i != j],
        dimen=2
    )

    # Parameters
    model.distance = pyo.Param(model.EDGES,
        initialize=lambda m, i, j: distance_matrix[(i, j)]
    )

    # Decision vars
    model.x = pyo.Var(model.EDGES, domain=pyo.Binary)  # Driving
    model.x_flight = pyo.Var(model.EDGES, domain=pyo.Binary)  # Flying
    model.y = pyo.Var(model.CITIES, domain=pyo.Binary)  # City visited
    model.days = pyo.Var(model.CITIES, domain=pyo.NonNegativeIntegers)
    model.u = pyo.Var(model.CITIES, domain=pyo.NonNegativeIntegers, bounds=(0, len(model.CITIES)))
    # fuel consumed arriving at each city (only when driving)
    model.f = pyo.Var(model.CITIES, domain=pyo.NonNegativeReals)

    # Venue selection
    venue_list = [(city.name, v.name) for city in cities for v in city.venues]
    model.VENUES = pyo.Set(initialize=venue_list, dimen=2)
    model.z = pyo.Var(model.VENUES, domain=pyo.Binary)

    # Constraints
    # --- Transportation feasibility rules ---
    def drive_allowed_rule(m, i, j):
        # No driving on long legs (force flight if that arc is used)
        if m.distance[i, j] > FLIGHT_DISTANCE_THRESHOLD:
            return m.x[i, j] == 0
        # Keep your general driving max-distance limiter too
        if m.distance[i, j] > MAX_DISTANCE:
            return m.x[i, j] == 0
        return pyo.Constraint.Skip
    model.drive_allowed = pyo.Constraint(model.EDGES, rule=drive_allowed_rule)

    def fly_allowed_rule(m, i, j):
        # No flights on short legs (force driving if that arc is used)
        if m.distance[i, j] <= FLIGHT_DISTANCE_THRESHOLD:
            return m.x_flight[i, j] == 0
        return pyo.Constraint.Skip
    model.fly_allowed = pyo.Constraint(model.EDGES, rule=fly_allowed_rule)

    # One mode max per edge
    def transport_mode_rule(m, i, j):
        return m.x[i, j] + m.x_flight[i, j] <= 1
    model.transport_mode = pyo.Constraint(model.EDGES, rule=transport_mode_rule)

    # Fuel usage definition - only for driven edges
    def fuel_use_rule(m, j):
        return m.f[j] == sum(
            m.distance[i, j] / FUEL_EFFICIENCY * m.x[i, j]
            for i in m.CITIES if i != j
        )
    model.fuel_use = pyo.Constraint(model.CITIES, rule=fuel_use_rule)
    
    # Tank capacity limiter (only matters when driving)
    if GAS_DIST_LIMITER:
        model.fuel_limit = pyo.Constraint(
            model.CITIES,
            rule=lambda m, j: m.f[j] <= FUEL_TANK_CAPACITY
        )

    # Routing - modified to handle both driving and flying
    def outflow_rule(m, i):
        total_outflow = sum(m.x[i, j] + m.x_flight[i, j] for j in m.CITIES if j != i)
        if i == HOME_BASE:
            return total_outflow == 1
        return total_outflow == m.y[i]
    model.outflow = pyo.Constraint(model.CITIES, rule=outflow_rule)

    def inflow_rule(m, j):
        total_inflow = sum(m.x[i, j] + m.x_flight[i, j] for i in m.CITIES if i != j)
        if j == HOME_BASE:
            return total_inflow == 1
        return total_inflow == m.y[j]
    model.inflow = pyo.Constraint(model.CITIES, rule=inflow_rule)

    def subtour_rule(m, i, j):
        if i != HOME_BASE and j != HOME_BASE and i != j:
            edge_used = m.x[i, j] + m.x_flight[i, j]
            return m.u[i] - m.u[j] + len(model.CITIES)*edge_used <= len(model.CITIES)-1
        return pyo.Constraint.Skip
    model.subtour = pyo.Constraint(model.EDGES, rule=subtour_rule)

    # City-day linkage
    model.day_visit = pyo.Constraint(model.CITIES, rule=lambda m,c: m.days[c] >= m.y[c])

    # Venue-city constraints
    model.one_venue = pyo.Constraint(
        model.CITIES,
        rule=lambda m,c: sum(m.z[c,v] for (cc,v) in m.VENUES if cc==c) <= 1
    )
    model.venue_link = pyo.Constraint(model.VENUES, rule=lambda m,c,v: m.z[c,v] <= m.y[c])
    model.pop_threshold = pyo.Constraint(
        model.VENUES,
        rule=lambda m,c,v: (m.z[c,v] == 0)
            if BAND_POPULARITY < next(vn for vn in city_data[c].venues if vn.name==v).popularity_threshold
            else pyo.Constraint.Skip
    )

    # Tour days
    model.total_days = pyo.Constraint(expr=sum(model.days[c] for c in model.CITIES) <= MAX_TOUR_DAYS)

    # Minimum stops constraint
    model.min_stops = pyo.Constraint(expr=sum(model.y[c] for c in model.CITIES) >= MIN_STOPS)

    # Cost expressions
    # Toll and parking only apply when driving
    model.toll_parking_total = pyo.Expression(
        expr=sum(
            model.x[i,j]*(city_data[j].toll_costs+city_data[j].parking_costs)
            for (i,j) in model.EDGES
        )
    )

    model.refuel_total = pyo.Expression(
    expr=sum(model.f[c] * city_data[c].avg_gas_price * NUM_VEHICLES for c in model.CITIES)
)


    model.staying_total = pyo.Expression(
        expr=sum(
            model.days[c]*(city_data[c].avg_hotel_cost+city_data[c].avg_meal_cost)
            for c in model.CITIES if c!=HOME_BASE
        )
    )

    # Flight costs
    model.flight_total = pyo.Expression(
        expr=sum(
            model.x_flight[i, j] * AVG_FLIGHT_COST_PER_PERSON * NUM_BAND_MEMBERS
            for (i, j) in model.EDGES
        )
    )

    # Budget
    total_costs = (model.toll_parking_total + model.refuel_total + 
                   model.staying_total + model.flight_total)
    model.budget = pyo.Constraint(expr=total_costs <= TOUR_BUDGET)

    # Objective
    def obj_rule(m):
        rev = sum(
            m.z[c,v] * next(vn for vn in city_data[c].venues if vn.name==v)
                .expected_revenue(city_data[c].regional_fanbase_strength)
            for (c,v) in m.VENUES
        )
        return rev - total_costs
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return model, model.toll_parking_total, model.refuel_total, model.staying_total, model.flight_total





# ============================================================================
# SOLVING AND RESULT ANALYSIS
# ============================================================================
def solve_model(model, solver_name="cbc"):
    solver = pyo.SolverFactory(solver_name)

    # Stop after 5 minutes
    # solver.options["seconds"] = 300

    # Allow 1% optimality gap (good enough for tour routing)
    # solver.options["ratio"] = 0.005   # CBC relative mip gap

    return solver.solve(model, tee=True)



def display_results(model, city_data, toll_parking_total, refuel_total, staying_total, flight_total):
    """
    Display the optimization results.

    Parameters:
        model (pyo.ConcreteModel): Solved model
        city_data (dict): Dictionary mapping city names to City objects
        toll_parking_total (pyo.Expression): Expression for total toll and parking costs
        refuel_total (pyo.Expression): Expression for total refueling costs
        staying_total (pyo.Expression): Expression for total accommodation costs
        flight_total (pyo.Expression): Expression for total flight costs
    """
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)

    # Display selected tour route (both driving and flying)
    print("\nTour Route:")
    print("  Driving legs:")
    driving_count = 0
    for (i, j) in model.EDGES:
        if pyo.value(model.x[i, j]) > 0.5:
            print(f"    {i} -> {j} (DRIVE)")
            driving_count += 1
    
    print("  Flying legs:")
    flying_count = 0
    for (i, j) in model.EDGES:
        if pyo.value(model.x_flight[i, j]) > 0.5:
            dist = pyo.value(model.distance[i, j])
            print(f"    {i} -> {j} (FLY - {dist} miles)")
            flying_count += 1
    
    if flying_count == 0:
        print("    (None - all legs driven)")

    # Display visited cities
    print("\nCities visited:")
    for c in model.CITIES:
        if pyo.value(model.y[c]) > 0.5:
            print(f"  {c}")

    # Display days spent in each city
    print("\nDays spent in each city:")
    for c in model.CITIES:
        days = pyo.value(model.days[c])
        if days > 0:
            print(f"  {c}: {days} days")

    # Display selected venues and their expected revenue
    print("\nSelected Venues:")
    for (c, v) in model.VENUES:
        if pyo.value(model.z[c, v]) > 0.5:
            venue_obj = next(venue for venue in city_data[c].venues if venue.name == v)
            revenue = venue_obj.expected_revenue(city_data[c].regional_fanbase_strength)
            print(f"  {c}: {v} (Expected Revenue: ${revenue:.2f})")

    # Display fuel levels
    if GAS_DIST_LIMITER:
        print("\nFuel consumed on arrival (gallons):")
        for c in model.CITIES:
            if pyo.value(model.y[c]) > 0.5 or c == HOME_BASE:
                fuel = pyo.value(model.f[c])
                if fuel > 0:
                    print(f"  {c}: {fuel:.2f}")

    # Display cost summary
    total_tp_cost = pyo.value(toll_parking_total)
    total_rf_cost = pyo.value(refuel_total)
    total_st_cost = pyo.value(staying_total)
    total_fl_cost = pyo.value(flight_total)
    total_expenses = total_tp_cost + total_rf_cost + total_st_cost + total_fl_cost

    print("\nCost Summary:")
    print(f"  Total Toll+Parking Cost: ${total_tp_cost:.2f}")
    print(f"  Total Refuel Cost: ${total_rf_cost:.2f}")
    print(f"  Total Staying Cost: ${total_st_cost:.2f}")
    print(f"  Total Flight Cost: ${total_fl_cost:.2f}")
    print(f"  Total Expenses: ${total_expenses:.2f}")
    print(f"  Net Profit: ${pyo.value(model.obj):.2f}")
    
    print(f"\nTransportation Summary:")
    print(f"  Driving legs: {driving_count}")
    print(f"  Flying legs: {flying_count}")


def reconstruct_tour_route(model):
    """
    Reconstruct the ordered tour route.

    Parameters:
        model (pyo.ConcreteModel): Solved model

    Returns:
        list: Ordered list of (from_city, to_city, mode) tuples where mode is 'DRIVE' or 'FLY'
    """
    # Build dictionary of selected arcs with their mode
    arc_dict = {}
    for (i, j) in model.EDGES:
        if pyo.value(model.x[i, j]) > 0.5:
            arc_dict[i] = (j, 'DRIVE')
        elif pyo.value(model.x_flight[i, j]) > 0.5:
            arc_dict[i] = (j, 'FLY')

    # Reconstruct ordered route
    ordered_route = []
    current_city = HOME_BASE
    while True:
        next_info = arc_dict.get(current_city)
        if next_info is None:
            break
        next_city, mode = next_info
        ordered_route.append((current_city, next_city, mode))
        if next_city == HOME_BASE:
            break
        current_city = next_city

    # Display the route
    print("\nOrdered Tour:")
    if ordered_route:
        route_str = ""
        for i, (from_city, to_city, mode) in enumerate(ordered_route):
            if i == 0:
                route_str += from_city
            route_str += f" --[{mode}]--> {to_city}"
        print(route_str)
    else:
        print("No feasible tour route found.")

    return ordered_route


def simulate_tour(model, ordered_route, city_data):
    """
    Simulate the tour to show budget flow and earnings for each leg.

    Parameters:
        model (pyo.ConcreteModel): Solved model
        ordered_route (list): Ordered list of (from_city, to_city, mode) tuples
        city_data (dict): Dictionary mapping city names to City objects
    """

    print("\n" + "="*110)
    print("TOUR SIMULATION (Budget Flow and Earnings)")
    print("="*110)

    initial_budget = TOUR_BUDGET
    cumulative_cost = 0.0
    cumulative_revenue = 0.0
    current_budget = initial_budget

    # Helper functions
    def get_toll_parking(city_name, mode):
        if mode == 'FLY':
            return 0.0
        return city_data[city_name].toll_costs + city_data[city_name].parking_costs

    def get_refuel_cost(city_name, mode):
        if mode == 'FLY':
            return 0.0
        fuel_consumed = pyo.value(model.f[city_name])
        return fuel_consumed * city_data[city_name].avg_gas_price
    
    def get_flight_cost(mode):
        if mode == 'FLY':
            return AVG_FLIGHT_COST_PER_PERSON * NUM_BAND_MEMBERS
        return 0.0

    def get_staying_cost(city_name):
        if city_name == HOME_BASE:
            return 0.0
        days = pyo.value(model.days[city_name])
        return days * (city_data[city_name].avg_hotel_cost + city_data[city_name].avg_meal_cost)

    def get_city_revenue(city_name):
        for (c, v) in model.VENUES:
            if c == city_name and pyo.value(model.z[c, v]) > 0.5:
                venue_obj = next(venue for venue in city_data[c].venues if venue.name == v)
                return venue_obj.expected_revenue(city_data[c].regional_fanbase_strength)
        return 0.0

    def get_selected_venue(city_name):
        for (c, v) in model.VENUES:
            if c == city_name and pyo.value(model.z[c, v]) > 0.5:
                return v
        return "-"

    # Display header
    print("\nLeg-by-Leg Financial Breakdown:")
    print("-" * 150)
    print(f"{'From':<15} {'To':<15} {'Mode':<6} {'Venue':<35} {'T+P':<8} {'Fuel':<8} {'Flight':<8} "
          f"{'Stay':<8} {'LegCost':<8} {'Budget':<10} {'Revenue':<10} {'Cum.Rev.':<10}")
    print("-" * 150)

    prev_city = HOME_BASE
    for (from_city, to_city, mode) in ordered_route:
        toll_cost = get_toll_parking(to_city, mode)
        refuel_cost = get_refuel_cost(to_city, mode)
        flight_cost = get_flight_cost(mode)
        stay_cost = get_staying_cost(to_city)
        leg_cost = toll_cost + refuel_cost + flight_cost + stay_cost
        cumulative_cost += leg_cost
        current_budget = initial_budget - cumulative_cost
        city_rev = get_city_revenue(to_city)
        cumulative_revenue += city_rev
        venue_name = get_selected_venue(to_city)

        print(f"{prev_city:<15} {to_city:<15} {mode:<6} {venue_name:<35} "
              f"${toll_cost:<7.2f} ${refuel_cost:<7.2f} ${flight_cost:<7.2f} ${stay_cost:<7.2f} "
              f"${leg_cost:<7.2f} ${current_budget:<9.2f} ${city_rev:<9.2f} ${cumulative_revenue:<9.2f}")

        prev_city = to_city

    print("-" * 150)
    print(f"\nFinal Remaining Budget: ${current_budget:.2f}")
    print(f"Total Cumulative Revenue: ${cumulative_revenue:.2f}")
    print(f"Net Profit (Revenue - Total Expenses): ${cumulative_revenue - cumulative_cost:.2f}")



def generate_tour_timeline(model, ordered_route):
    """
    Generate a timeline of the tour schedule.

    Parameters:
        model (pyo.ConcreteModel): Solved model
        ordered_route (list): Ordered list of (from_city, to_city, mode) tuples
    """
    print("\n" + "="*80)
    print("TOUR SCHEDULE TIMELINE")
    print("="*80)

    if not ordered_route:
        print("No feasible tour route found. Cannot generate timeline.")
        return

    # For this timeline, we assume the tour starts on day 1
    current_day = 1

    # Construct a list of visited cities in order
    visited_cities = [(ordered_route[0][0], 'START')]
    for (_, to_city, mode) in ordered_route:
        visited_cities.append((to_city, mode))

    for city, mode in visited_cities:
        arrival_day = current_day
        days_spent = pyo.value(model.days[city])
        
        mode_str = f" (arrived via {mode})" if mode != 'START' else " (home base)"
        print(f"{city} visited on day {arrival_day}{mode_str} (staying for {days_spent} day(s))")

        # Check if there's a performance at this city
        for (c, v) in model.VENUES:
            if c == city and pyo.value(model.z[c, v]) > 0.5:
                # Assume the concert is played on the arrival day
                print(f"  Performance at '{v}' on day {arrival_day}\n")

        current_day += days_spent


# Function to create and display the tour map
def visualize_tour_route(ordered_route, city_data, city_venues=None, save_path=None):
    """
    Visualize a tour route on a map.

    Parameters:
        ordered_route (list): List of (from_city, to_city, mode) tuples representing the route
        city_data (dict): Dictionary with city data including coordinates
        city_venues (dict, optional): Dictionary mapping cities to their selected venues
        save_path (str, optional): Path to save the figure
    """
    # Determine if US or European tour
    city_coords = create_city_coords()

    # Create a directed graph for the tour
    G = nx.DiGraph()

    # Extract cities from ordered route
    cities_in_route = []
    for from_city, to_city, mode in ordered_route:
        if from_city not in cities_in_route:
            cities_in_route.append(from_city)
        if to_city not in cities_in_route:
            cities_in_route.append(to_city)
        G.add_edge(from_city, to_city)

    # ---- INTERNATIONAL FIX: center projection + extent based on actual route ----
    # (keeps the same visual "type" and styling, but ensures international/mixed tours are visible)
    route_lons = []
    route_lats = []
    for c in cities_in_route:
        if c in city_coords:
            lon, lat = city_coords[c]
            route_lons.append(lon)
            route_lats.append(lat)

    if route_lons and route_lats:
        min_lon, max_lon = min(route_lons), max(route_lons)
        min_lat, max_lat = min(route_lats), max(route_lats)
        central_longitude = (min_lon + max_lon) / 2
        central_latitude = (min_lat + max_lat) / 2
    else:
        # Fallback (matches old behavior)
        central_longitude = -95
        central_latitude = 37.5
        min_lon, max_lon, min_lat, max_lat = -125, -66.5, 24, 50

    # Create figure with cartopy projection (same projection type; now centered on the route)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(
        1, 1, 1,
        projection=ccrs.LambertConformal(
            central_longitude=central_longitude,
            central_latitude=central_latitude
        )
    )

    # Set map extent
    if route_lons and route_lats:
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat

        pad_lon = max(5, lon_span * 0.15)
        pad_lat = max(5, lat_span * 0.15)

        ax.set_extent(
            [min_lon - pad_lon, max_lon + pad_lon, min_lat - pad_lat, max_lat + pad_lat],
            ccrs.PlateCarree()
        )
    else:
        # Old extent logic (kept for safety)
        if "London" in [c for c in city_data.keys()]:
            ax.set_extent([-11, 20, 35, 60], ccrs.PlateCarree())
        else:
            ax.set_extent([-125, -66.5, 24, 50], ccrs.PlateCarree())

    # Add map features (unchanged)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    # STATES only makes sense in/near the US; keep it where it won't break international maps
    try:
        ax.add_feature(cfeature.STATES, linestyle=':')
    except Exception:
        pass

    # Plot cities
    for city, (lon, lat) in city_coords.items():
        if city in cities_in_route:
            # Add city name
            ax.text(lon + 0.5, lat, city, transform=ccrs.PlateCarree(),
                    fontsize=9, ha='left', va='center', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

            if city == HOME_BASE:
                # Home base is a black star
                ax.plot(lon, lat, '*', transform=ccrs.PlateCarree(),
                        markersize=12, color='black', alpha=1, zorder=3)
            else:
                # Cities in route are red dots
                ax.plot(lon, lat, 'o', transform=ccrs.PlateCarree(),
                        markersize=8, color='red', alpha=0.5, zorder=3)
        else:
            # Cities not in route are gray
            ax.plot(lon, lat, 'o', transform=ccrs.PlateCarree(),
                    markersize=12, color='gray', alpha=0.5, zorder=3)

    # Draw edges with different colors for driving vs flying
    for i, (from_city, to_city, mode) in enumerate(ordered_route):
        from_lon, from_lat = city_coords[from_city]
        to_lon, to_lat = city_coords[to_city]

        # Choose color based on mode
        color = 'red' if mode == 'DRIVE' else 'blue'
        linestyle = '-' if mode == 'DRIVE' else '--'

        # Draw the edge
        ax.plot([from_lon, to_lon], [from_lat, to_lat], linestyle,
                transform=ccrs.PlateCarree(), linewidth=2,
                color=color, zorder=4)

        # Add an arrow to show direction
        mid_lon = (from_lon + to_lon) / 2
        mid_lat = (from_lat + to_lat) / 2
        dx = to_lon - from_lon
        dy = to_lat - from_lat
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        dx = dx / length
        dy = dy / length

        ax.arrow(mid_lon - dx * 0.2, mid_lat - dy * 0.2,
                 dx * 0.4, dy * 0.4,
                 transform=ccrs.PlateCarree(),
                 head_width=0.3, head_length=0.3,
                 fc='black', ec='black', zorder=4)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='red', label='Cities in Tour',
               markerfacecolor='red', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='gray', label='Other Cities',
               markerfacecolor='gray', markersize=6, alpha=0.5, linestyle='None'),
        Line2D([0], [0], color='red', linewidth=2, label='Driving'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Flying')
    ]

    # Add venues to the legend if provided
    if city_venues:
        venue_info = []
        for city in cities_in_route:
            if city in city_venues and city_venues[city]:
                venue_info.append(f"{city}: {city_venues[city]}")

        if venue_info:
            venue_text = "\n".join(venue_info)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax.text(0.05, 0.05, "Venues:\n" + venue_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='bottom', bbox=props)

    ax.legend(handles=legend_elements, loc='lower right')

    # Add title and grid
    plt.title('Optimized Tour Route', fontsize=16)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig, ax



# Function to extract data from model results
def extract_tour_data(model):
    """
    Extract tour data from solved model.
    
    Parameters:
        model (pyo.ConcreteModel): Solved model
    
    Returns:
        tuple: (ordered_route, city_venues)
    """
    # Build a dict of all chosen arcs with mode
    arc_dict = {}
    for (i, j) in model.EDGES:
        if pyo.value(model.x[i, j]) > 0.5:
            arc_dict[i] = (j, 'DRIVE')
        elif pyo.value(model.x_flight[i, j]) > 0.5:
            arc_dict[i] = (j, 'FLY')

    # **Force-start** from the real global HOME_BASE
    current_city = HOME_BASE
    ordered_route = []

    # Walk the route until you get back or run out of arcs
    while True:
        next_info = arc_dict.get(current_city)
        if next_info is None:
            break
        next_city, mode = next_info
        ordered_route.append((current_city, next_city, mode))
        if next_city == HOME_BASE:
            break
        current_city = next_city

    # Grab whichever venues you picked
    city_venues = {
        c: v for (c, v) in model.VENUES
        if pyo.value(model.z[c, v]) > 0.5
    }

    return ordered_route, city_venues



# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    print("running...")
    # Initialize data
    cities, city_data, city_names = create_cities_and_venues()
    distance_matrix = create_distance_matrix(city_names)

    # Create and solve model
    model, toll_parking_total, refuel_total, staying_total, flight_total = create_optimization_model(
        cities, city_data, distance_matrix
    )

    solver = 'cbc'  # Path to solver
    results = solve_model(model, solver)

    # Check if solution is feasible
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("The model is infeasible.")
        return

    else:
        # Display and analyze results
        display_results(model, city_data, toll_parking_total, refuel_total, staying_total, flight_total)
        ordered_route = reconstruct_tour_route(model)
        simulate_tour(model, ordered_route, city_data)
        generate_tour_timeline(model, ordered_route)
        ordered_route, city_venues = extract_tour_data(model)
        visualize_tour_route(ordered_route, city_data, city_venues)



if __name__ == "__main__":
    main()