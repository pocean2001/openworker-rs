//! Weather skill — a [`Tool`] that answers "what's the weather?" for a city, a
//! `lat,lon` coordinate pair, or the caller's own location.
//!
//! It is backed by the free [Open-Meteo](https://open-meteo.com) APIs (geocoding +
//! forecast), which need **no API key**, and by free IP-geolocation services for the
//! "auto-detect my location" path. The returned JSON is deliberately structured (not a
//! pre-rendered string) so the model can quote it, format it, or feed it into further
//! tool calls.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::permissions::{RiskLevel, ToolMetadata};
use crate::provider::{FunctionSpec, ToolSpec};
use crate::tools::Tool;

/// Get current weather (and a short forecast) for a location.
pub struct GetWeather;

#[async_trait]
impl Tool for GetWeather {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "get_weather".into(),
                description: "Get the current weather and a short forecast for a location. \
                              Pass a city name (e.g. '北京', 'Tokyo', 'San Francisco'), a \
                              'latitude,longitude' pair, or nothing to auto-detect the caller's \
                              location from its IP. Uses the free Open-Meteo API, no key needed."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, or 'latitude,longitude' (e.g. '39.9042,116.4074'). Omit to auto-detect location."
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "description": "metric = °C / km/h, imperial = °F / mph. Default: metric."
                        },
                        "forecast_days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 7,
                            "description": "Number of daily forecast days to include (1-7). Default: 1."
                        }
                    }
                }),
            },
        }
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            category: "weather".into(),
            risk_level: RiskLevel::Low, // read-only network call: never needs approval
            ..Default::default()
        }
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let raw_loc = args
            .get("location")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let units = args
            .get("units")
            .and_then(|v| v.as_str())
            .unwrap_or("metric");
        let forecast_days = args
            .get("forecast_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .clamp(1, 7);

        // Resolve the location to coordinates + a human-readable label.
        let (lat, lon, label) = match &raw_loc {
            Some(loc) => resolve_named(&client, loc).await?,
            None => detect_location(&client).await?,
        };

        let (temp_unit, wind_unit) = if units == "imperial" {
            ("fahrenheit", "mph")
        } else {
            ("celsius", "kmh")
        };
        let url = format!(
            "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}\
             &current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,\
             precipitation,weather_code,wind_speed_10m,wind_direction_10m\
             &daily=weather_code,temperature_2m_max,temperature_2m_min,\
             precipitation_probability_max&timezone=auto&forecast_days={forecast_days}\
             &temperature_unit={temp_unit}&wind_speed_unit={wind_unit}"
        );
        let resp = client.get(&url).send().await?.error_for_status()?;
        let body: Value = resp.json().await?;

        let current = body.get("current").cloned().unwrap_or(Value::Null);
        let description = current
            .get("weather_code")
            .and_then(|v| v.as_i64())
            .map(|c| describe_weather_code(c as i32).to_string())
            .unwrap_or_else(|| "unknown".into());

        let mut forecast = Vec::new();
        if let Some(daily) = body.get("daily") {
            if let Some(dates) = daily.get("time").and_then(|t| t.as_array()) {
                let codes = daily.get("weather_code").and_then(|t| t.as_array());
                let tmax = daily.get("temperature_2m_max").and_then(|t| t.as_array());
                let tmin = daily.get("temperature_2m_min").and_then(|t| t.as_array());
                let pop = daily
                    .get("precipitation_probability_max")
                    .and_then(|t| t.as_array());
                for (i, date) in dates.iter().enumerate() {
                    let code = codes
                        .and_then(|a| a.get(i))
                        .and_then(|c| c.as_i64())
                        .unwrap_or(0) as i32;
                    forecast.push(json!({
                        "date": date.as_str().unwrap_or(""),
                        "weather": describe_weather_code(code),
                        "temperature_max": tmax.and_then(|a| a.get(i)).and_then(|v| v.as_f64()),
                        "temperature_min": tmin.and_then(|a| a.get(i)).and_then(|v| v.as_f64()),
                        "precipitation_probability": pop.and_then(|a| a.get(i)).and_then(|v| v.as_i64()),
                    }));
                }
            }
        }

        let cur = |k: &str| current.get(k).and_then(|v| v.as_f64());
        Ok(json!({
            "location": {
                "query": raw_loc.unwrap_or_else(|| "auto-detected".into()),
                "resolved": label,
                "latitude": lat,
                "longitude": lon,
                "units": units,
            },
            "current": {
                "description": description,
                "temperature": cur("temperature_2m"),
                "apparent_temperature": cur("apparent_temperature"),
                "relative_humidity": cur("relative_humidity_2m"),
                "precipitation": cur("precipitation"),
                "wind_speed": cur("wind_speed_10m"),
                "wind_direction": cur("wind_direction_10m"),
                "is_day": current.get("is_day").and_then(|v| v.as_i64()).map(|v| v == 1),
                "time": current.get("time").and_then(|v| v.as_str()),
            },
            "forecast": forecast,
        }))
    }
}

/// Resolve a user-provided location: a `lat,lon` string is used directly, anything else
/// goes through the Open-Meteo geocoding API.
async fn resolve_named(client: &reqwest::Client, loc: &str) -> Result<(f64, f64, String)> {
    if let Some((a, b)) = loc.split_once(',') {
        if let (Ok(a), Ok(b)) = (a.trim().parse::<f64>(), b.trim().parse::<f64>()) {
            return Ok((a, b, format!("{a},{b}")));
        }
    }
    geocode(client, loc).await
}

/// Resolve a place name to coordinates via the Open-Meteo geocoding API.
async fn geocode(client: &reqwest::Client, name: &str) -> Result<(f64, f64, String)> {
    let url = format!(
        "https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=zh&format=json",
        urlencode(name)
    );
    let resp = client.get(&url).send().await?.error_for_status()?;
    let body: Value = resp.json().await?;
    let first = body
        .get("results")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| anyhow!("could not geocode location '{}'", name))?;
    let lat = first
        .get("latitude")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow!("geocoding returned no latitude"))?;
    let lon = first
        .get("longitude")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow!("geocoding returned no longitude"))?;
    // Build "Name, Admin1, Country" as the resolved label.
    let label = [
        first.get("name").and_then(|v| v.as_str()),
        first.get("admin1").and_then(|v| v.as_str()),
        first.get("country").and_then(|v| v.as_str()),
    ]
    .iter()
    .filter_map(|s| *s)
    .collect::<Vec<_>>()
    .join(", ");
    Ok((lat, lon, if label.is_empty() { name.to_string() } else { label }))
}

/// Best-effort IP-based location detection. Tries a couple of free, keyless services;
/// the first that answers with coordinates wins.
async fn detect_location(client: &reqwest::Client) -> Result<(f64, f64, String)> {
    let endpoints = [
        "https://ipwho.is/",
        "http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country",
    ];
    for url in endpoints {
        if let Ok(resp) = client.get(url).send().await {
            if let Ok(v) = resp.json::<Value>().await {
                let lat = v
                    .get("latitude")
                    .and_then(|x| x.as_f64())
                    .or_else(|| v.get("lat").and_then(|x| x.as_f64()));
                let lon = v
                    .get("longitude")
                    .and_then(|x| x.as_f64())
                    .or_else(|| v.get("lon").and_then(|x| x.as_f64()));
                if let (Some(lat), Some(lon)) = (lat, lon) {
                    let label = [
                        v.get("city").and_then(|x| x.as_str()),
                        v.get("regionName").and_then(|x| x.as_str()),
                        v.get("country").and_then(|x| x.as_str()),
                    ]
                    .iter()
                    .filter_map(|s| *s)
                    .collect::<Vec<_>>()
                    .join(", ");
                    let label = if label.is_empty() {
                        format!("{lat:.4},{lon:.4}")
                    } else {
                        label
                    };
                    return Ok((lat, lon, label));
                }
            }
        }
    }
    Err(anyhow!(
        "could not auto-detect location; pass a city name or 'lat,lon' explicitly"
    ))
}

/// WMO weather code → human-readable description (English; the model can translate it).
fn describe_weather_code(code: i32) -> &'static str {
    match code {
        0 => "Clear sky",
        1 => "Mainly clear",
        2 => "Partly cloudy",
        3 => "Overcast",
        45 | 48 => "Fog",
        51 => "Light drizzle",
        53 => "Drizzle",
        55 => "Dense drizzle",
        56 | 57 => "Freezing drizzle",
        61 => "Slight rain",
        63 => "Rain",
        65 => "Heavy rain",
        66 | 67 => "Freezing rain",
        71 => "Slight snow",
        73 => "Snow",
        75 => "Heavy snow",
        77 => "Snow grains",
        80 => "Slight rain showers",
        81 => "Rain showers",
        82 => "Violent rain showers",
        85 => "Slight snow showers",
        86 => "Snow showers",
        95 => "Thunderstorm",
        96 | 99 => "Thunderstorm with hail",
        _ => "Unknown",
    }
}

/// Percent-encode a string for use in a URL query value (UTF-8 aware).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_code_descriptions_cover_wmo_ranges() {
        assert_eq!(describe_weather_code(0), "Clear sky");
        assert_eq!(describe_weather_code(63), "Rain");
        assert_eq!(describe_weather_code(95), "Thunderstorm");
        assert_eq!(describe_weather_code(999), "Unknown");
    }

    #[test]
    fn urlencode_handles_cjk_and_spaces() {
        assert_eq!(urlencode("北京"), "%E5%8C%97%E4%BA%AC");
        assert_eq!(urlencode("San Francisco"), "San+Francisco");
        assert_eq!(urlencode("a b&c"), "a+b%26c");
    }
}
