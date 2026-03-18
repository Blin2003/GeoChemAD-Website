# GeoChemAD v2

GeoChemAD v2 extends the original MVP map viewer into a browser-based geochemical anomaly visualization tool.

## What it does

- loads **sample CSV** and **site CSV** separately
- detects analyte columns such as `Au_ppm`, `Cu_ppm`, `Fe_ppm`, `W_ppm`, `Ni_ppm`
- preprocesses values in the browser
  - filters values at or below the invalid threshold, default `-9999`
  - replaces non-positive values with half of the smallest positive value for that element
  - optionally applies `log(1 + x)` transformation
- computes anomaly scores
  - Z-score
  - Robust Z-score using median and MAD
  - Percentile score
  - Composite mean across multiple selected elements
  - Composite max across multiple selected elements
- colors sample points by anomaly score
- highlights points above the anomaly threshold
- displays nearest known site distance for each scored sample point

## Run locally

```bash
npm install
npm start
```

Then open the local URL shown in the terminal.

## Expected files

### Sample CSV
Must contain:
- `X` longitude
- `Y` latitude
- one or more analyte columns such as `Au_ppm`, `Cu_ppm`, `W_ppm`

### Site CSV
Must contain:
- `X` longitude
- `Y` latitude
- preferably `SITE_CODE`, `SITE_TITLE`, or related site fields

## Suggested workflow

1. Upload a sample CSV.
2. Upload the matching site CSV.
3. Choose a primary element or multiple elements.
4. Keep `log(1+x)` enabled for most geochemical data.
5. Start with **Robust Z-score** and threshold `2.0`.
6. Toggle **show only anomaly points** to inspect targets.

## Notes

- this is still a **pure front-end** project with no model training backend
- it is designed as a strong practical bridge between your current MVP and a later machine-learning version
- it works best as an interactive anomaly exploration tool for the provided GeoChemAD-style datasets
