from typing import Union
import numpy as np
from fastapi import FastAPI
from scipy.stats import norm 
import math

app = FastAPI()

# Black-Scholes Formula
# C=SN(d1) - Ke^-rtN(d2) WHERE d1 = (ln(S/K) + (r + (σ^2)/2)t) / σ√t and d2 = d1 - σ√t
# C = Call Option Price
# S = Current Stock Price
# K = Strike Price
# r = Risk-Free Rate
# t = Time to Expiration
# N = A Normal Distribution
# σ = Volatility
# This is the formula we will be using to calculate the option price. There exists several input variables that the API will handle in the functions below. The input will have to be sent as a JSON object to the API. 
# The API will then return the calculated option price as a JSON object.

#This is the boiler-plate, but as we will dynamically display the option price, there will be a range of calculations performed.
def black_scholes(S, K, r, t, sigma, option_type):
    d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return "Invalid option type"

#This function will process a range of data to help us visualize the option price's dependency on strike price and asset price.
# Original range endpoint for stock price only
@app.get("/black_scholes/range_asset_vs_strike")
async def calculate_option_range(
    option_type: str, 
    S: float,  # Base stock price
    K: float,  # Base strike price
    r: float, 
    t: float, 
    sigma: float,
    points: int = 10,  # Number of points per dimension
    range_multiplier: float = 2.0
):
    try:
        # Create arrays for both stock and strike prices
        prices = np.linspace(min(S, K)/2, max(S, K)*range_multiplier, points)
        
        # Calculate options for all combinations
        option_prices = [
            {
                "stock_price": float(stock_price),
                "strike_price": float(strike_price),
                "option_price": black_scholes(stock_price, strike_price, r, t, sigma, option_type)
            }
            for stock_price in prices
            for strike_price in prices
        ]
        
        return {
            "parameters": {
                "option_type": option_type,
                "base_stock_price": S,
                "base_strike_price": K,
                "risk_free_rate": r,
                "time_to_expiry": t,
                "volatility": sigma
            },
            "price_range": option_prices
        }
    except ValueError as e:
        return {"error": str(e)}

@app.get("/black_scholes/range_asset_vs_IV")
async def calculate_option_range(
    option_type: str, 
    S: float,  # Base stock price
    K: float,  # Base strike price
    r: float, 
    t: float, 
    sigma: float,
    points: int = 10,  # Number of points per dimension
    range_multiplier: float = 2.0
):
    try:
        # Create arrays for both stock and strike prices
        prices = np.linspace(min(S, K)/2, max(S, K)*range_multiplier, points)
        volatilities = np.linspace(0.01, 1, points)  # IV from 1% to 100%
        
        # Calculate options for all combinations
        option_prices = [
            {
                "stock_price": float(stock_price),
                "IV": float(iv),
                "option_price": black_scholes(stock_price, K, r, t, iv, option_type)
            }
            for stock_price in prices
            for iv in volatilities]
        
        return {
            "parameters": {
                "option_type": option_type,
                "base_stock_price": S,
                "base_strike_price": K,
                "risk_free_rate": r,
                "time_to_expiry": t,
                "volatility": sigma
            },
            "price_range": option_prices
        }
    except ValueError as e:
        return {"error": str(e)}

@app.get("/black_scholes/range_asset_vs_ttm")
async def calculate_option_range(
    option_type: str, 
    S: float,  # Base stock price
    K: float,  # Base strike price
    r: float, 
    t: float, 
    sigma: float,
    points: int = 10,  # Number of points per dimension
    range_multiplier: float = 2.0
):
    try:
        # Create arrays for both stock and strike prices
        prices = np.linspace(min(S, K)/2, max(S, K)*range_multiplier, points)
        ttm = np.linspace(0, 10, points, dtype=int)  # Time to maturity in integer days
        
        # Calculate options for all combinations
        option_prices = [
            {
                "stock_price": float(stock_price),
                "ttm": float(days),
                "option_price": black_scholes(stock_price, K, r, days, sigma, option_type)
            }
            for stock_price in prices
            for days in ttm]
        
        return {
            "parameters": {
                "option_type": option_type,
                "base_stock_price": S,
                "base_strike_price": K,
                "risk_free_rate": r,
                "time_to_expiry": t,
                "volatility": sigma
            },
            "price_range": option_prices
        }
    except ValueError as e:
        return {"error": str(e)}



# Single price endpoint
@app.get("/black_scholes")
async def calculate_option(
    option_type: str, 
    S: float, 
    K: float, 
    r: float, 
    t: float, 
    sigma: float
):
    try:
        price = black_scholes(S, K, r, t, sigma, option_type)
        return {
            "option_type": option_type,
            "stock_price": S,
            "strike_price": K,
            "risk_free_rate": r,
            "time_to_expiry": t,
            "volatility": sigma,
            "option_price": price
        }
    except ValueError as e:
        return {"error": str(e)}