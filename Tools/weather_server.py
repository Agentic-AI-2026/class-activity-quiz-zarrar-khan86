# ============================================================
# weather_server.py
# REMOTE MCP SERVER — Real-time weather using Open-Meteo API
# Run in a separate terminal: python weather_server.py
# Keep that terminal open while using your notebook!
# ============================================================

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

CITY_COORDS = {
    "london":     (51.5074, -0.1278),
    "paris":      (48.8566,  2.3522),
    "new york":   (40.7128, -74.0060),
    "tokyo":      (35.6762, 139.6503),
    "karachi":    (24.8607,  67.0011),
    "lahore":     (31.5204,  74.3587),
    "islamabad":  (33.6844,  73.0479),
    "rawalpindi": (33.5651,  73.0169),
    "dubai":      (25.2048,  55.2708),
    "berlin":     (52.5200,  13.4050),
    "sydney":    (-33.8688, 151.2093),
    "chicago":    (41.8781, -87.6298),
}


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get real-time current weather for a city using Open-Meteo free API.
    Returns temperature, wind speed, humidity, and sky condition.
    Available: London, Paris, New York, Tokyo, Karachi, Lahore,
    Islamabad, Rawalpindi, Dubai, Berlin, Sydney, Chicago."""
    coords = CITY_COORDS.get(city.lower().strip())
    if not coords:
        available = ", ".join(c.title() for c in CITY_COORDS)
        return f"City '{city}' not found. Available cities: {available}"

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relativehumidity_2m,apparent_temperature"
    )
    try:
        data     = requests.get(url, timeout=5).json()
        cw       = data.get("current_weather", {})
        temp     = cw.get("temperature", "N/A")
        wind     = cw.get("windspeed",   "N/A")
        wcode    = cw.get("weathercode",  0)
        cond     = "Sunny" if wcode < 3 else "Cloudy" if wcode < 50 else "Rainy"
        humidity = data.get("hourly", {}).get("relativehumidity_2m",  ["N/A"])[0]
        feels    = data.get("hourly", {}).get("apparent_temperature", ["N/A"])[0]
        return (
            f"Current weather in {city.title()}:\n"
            f"  Condition : {cond}\n"
            f"  Temp      : {temp} C\n"
            f"  Feels like: {feels} C\n"
            f"  Wind      : {wind} km/h\n"
            f"  Humidity  : {humidity}%"
        )
    except requests.Timeout:
        return f"Weather API timed out for '{city}'"
    except Exception as e:
        return f"Weather API error: {e}"


@mcp.tool()
def get_weather_forecast(city: str, days: int) -> str:
    """Get a weather forecast for a city for the next N days (1-7).
    Returns daily temperature range and general conditions."""
    if days < 1 or days > 7:
        return "Please provide a number of days between 1 and 7."

    coords = CITY_COORDS.get(city.lower().strip())
    if not coords:
        return f"City '{city}' not found."

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
        f"&forecast_days={days}"
        f"&timezone=auto"
    )
    try:
        data   = requests.get(url, timeout=5).json()
        daily  = data.get("daily", {})
        dates  = daily.get("time",                [])
        maxts  = daily.get("temperature_2m_max",  [])
        mints  = daily.get("temperature_2m_min",  [])
        wcodes = daily.get("weathercode",         [])

        lines = [f"Forecast for {city.title()} ({days} days):"]
        for i in range(min(days, len(dates))):
            wc   = wcodes[i] if i < len(wcodes) else 0
            cond = "Sunny" if wc < 3 else "Cloudy" if wc < 50 else "Rainy"
            lines.append(
                f"  {dates[i]} : {cond}, "
                f"High {maxts[i]}C / Low {mints[i]}C"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Forecast error: {e}"


# ============================================================
# START THE SERVER
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
