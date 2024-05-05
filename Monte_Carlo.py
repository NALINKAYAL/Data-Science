#!/usr/bin/env python
# coding: utf-8

# ## Monte Carlo Simulation
# 
# ## Group Members:  
# ## Nalin Kayal(211060020)
# ##                                Vansh Panchal(211060024)
# ##                                Ameya Jamkar (211060005)

# # TASK 1 : MONTE CARLOS SIMULATION

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def generate_points(num_points):
    # Generate random points within a square [-1, 1] x [-1, 1]
    points = np.random.rand(num_points, 2) * 2 - 1
    return points

def is_inside_circle(x, y):
    # Check if a point (x, y) is inside the circle (unit circle centered at the origin)
    return x**2 + y**2 <= 1

def is_inside_diamond(x, y):
    # Check if a point (x, y) is inside the diamond
    return (abs(x) + abs(y)) <= 1

def monte_carlo_simulation(num_points):
    points = generate_points(num_points)

    circle_points = sum(1 for x, y in points if is_inside_circle(x, y))
    diamond_points = sum(1 for x, y in points if is_inside_diamond(x, y))

    circle_area = circle_points / num_points * 4  # Area of the circle (pi * r^2, r=1)
    diamond_area = diamond_points / num_points * 4  # Area of the diamond (side length = 2)

    ratio_points = diamond_points / circle_points
    area_ratio = diamond_area / circle_area

    print(f"Estimated Area of Circle: {circle_area}")
    print(f"Estimated Area of Diamond: {diamond_area}")
    print(f"Ratio of Points in Diamond to Points in Circle: {ratio_points}")
    print(f"Ratio of Area of Diamond to Area of Circle: {area_ratio}")

    # Plot the points
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.5, label='Generated Points')
    plt.scatter([point[0] for point in points if is_inside_circle(point[0], point[1])],
                [point[1] for point in points if is_inside_circle(point[0], point[1])],
                c='green', alpha=0.5, label='Inside Circle')
    plt.scatter([point[0] for point in points if is_inside_diamond(point[0], point[1])],
                [point[1] for point in points if is_inside_diamond(point[0], point[1])],
                c='red', alpha=0.5, label='Inside Diamond')
    
    plt.title('Monte Carlo Simulation for Circle and Diamond')
    plt.legend()
    plt.show()

# Number of points in the simulation
num_points = 10000

# Run the Monte Carlo simulation
monte_carlo_simulation(num_points)


# In[3]:


# TASK 2 : Risk Analysis for Investment Portfolio Using Monte Carlo Simulation


# In[4]:


import random
import numpy as np

# Portfolio parameters
weights = [0.5, 0.8, 0.2]  # weights for stocks, bonds, and real estate
num_samples = 10000  # Number of Monte Carlo samples

# Asset characteristics
mean_returns = [0.08, -0.4, 0.06]
std_devs = [0.12, 0.08, 0.1]

# Initialize counter for negative portfolio returns
negative_returns_count = 0

# Perform Monte Carlo simulation
for _ in range(num_samples):
    # Generate random returns for each asset
    asset_returns = [random.gauss(mean, std_dev) for mean, std_dev in zip(mean_returns, std_devs)]

    # Calculate portfolio return
    portfolio_return = np.dot(weights, asset_returns)

    # Check if the portfolio has a negative return
    if portfolio_return < 0:
        negative_returns_count += 1

# Calculate the estimated probability of negative portfolio return
probability_negative_return = negative_returns_count / num_samples

print("Monte Carlo Estimated Probability of Negative Portfolio Return:", probability_negative_return)


# In[9]:


import random

def monte_carlo_draw_card(num_simulations=10000):
    # Define the deck of cards with different colors
    deck = ['red'] * 10 + ['blue'] * 15 + ['green'] * 5  # 30 cards total
    
    # Track the number of times a red card is drawn
    red_count = 0
    
    # Simulate drawing a card num_simulations times
    for _ in range(num_simulations):
        drawn_card = random.choice(deck)  # Randomly draw a card from the deck
        if drawn_card == 'red':
            red_count += 1
    
    # Calculate the probability of drawing a red card
    probability_red = red_count / num_simulations
    return probability_red

# Estimate the probability of drawing a red card
probability_red = monte_carlo_draw_card()
print(f"Estimated probability of drawing a red card: {probability_red:.4f}")


# In[10]:


import random

def monte_carlo_coin_flip(num_simulations=10000):
    # Track the number of times heads is flipped
    heads_count = 0
    
    # Simulate flipping a coin num_simulations times
    for _ in range(num_simulations):
        flip = random.choice(['heads', 'tails'])  # Randomly flip the coin
        if flip == 'heads':
            heads_count += 1
    
    # Calculate the probability of getting heads
    probability_heads = heads_count / num_simulations
    return probability_heads

# Estimate the probability of flipping heads
probability_heads = monte_carlo_coin_flip()
print(f"Estimated probability of flipping heads: {probability_heads:.4f}")


# In[11]:


import random

def monte_carlo_roll_die(num_simulations=10000, target_number=4):
    # Track the number of times the target number is rolled
    target_count = 0
    
    # Simulate rolling a die num_simulations times
    for _ in range(num_simulations):
        roll = random.randint(1, 6)  # Randomly roll the die (1 to 6)
        if roll == target_number:
            target_count += 1
    
    # Calculate the probability of rolling the target number
    probability_target = target_count / num_simulations
    return probability_target

# Estimate the probability of rolling a 4
probability_target = monte_carlo_roll_die()
print(f"Estimated probability of rolling a 4: {probability_target:.4f}")


# In[12]:


import random
import math

def monte_carlo_circle_area(radius=1, num_simulations=10000):
    # Initialize a counter for the number of points inside the circle
    points_inside_circle = 0
    
    # Define the bounding box side length (it's twice the radius)
    bounding_box_side = radius * 2
    bounding_box_area = bounding_box_side ** 2  # Area of the square bounding box
    
    # Simulate num_simulations random points within the bounding box
    for _ in range(num_simulations):
        # Generate a random point (x, y) within the bounding box
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        
        # Check if the point is inside the circle
        if x**2 + y**2 <= radius**2:
            points_inside_circle += 1
    
    # Calculate the proportion of points inside the circle
    proportion_inside_circle = points_inside_circle / num_simulations
    
    # Estimate the area of the circle using the proportion and the bounding box area
    estimated_circle_area = proportion_inside_circle * bounding_box_area
    
    return estimated_circle_area

# Estimate the area of a circle with radius 1
radius = 1
estimated_area = monte_carlo_circle_area(radius)
print(f"Estimated area of a circle with radius {radius}: {estimated_area:.4f}")

# You can compare the estimated area with the theoretical area using the formula pi * radius^2
theoretical_area = math.pi * radius ** 2
print(f"Theoretical area of a circle with radius {radius}: {theoretical_area:.4f}")

# Calculate the percentage error (optional)
percentage_error = abs((estimated_area - theoretical_area) / theoretical_area) * 100
print(f"Percentage error: {percentage_error:.2f}%")


# In[ ]:




