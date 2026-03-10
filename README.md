# GeoChemAD-Website

A minimal front-end project for viewing geochemical CSV data on a Leaflet map.

## Features
- Open the webpage locally
- Upload one CSV file at a time
- Supports both sample CSV and site CSV
- Automatically detects `X` / `Y` as longitude / latitude
- Displays points on a map
- Popup shows key fields only
- Ignores missing values such as `-9999`

## Files
- `index.html` - page layout
- `style.css` - page styles
- `main.js` - map setup, upload, CSV parsing, point rendering
- `package.json` - simple local start command

## Run
### Option 1: open directly
You can double-click `index.html`.

### Option 2: run a local static server (recommended)
In the project folder:

```bash
npx serve .
```

Then open the local address shown in the terminal.

## CSV requirements
The CSV must contain:
- `X` = longitude
- `Y` = latitude

If columns like `SITE_CODE` / `SITE_TITLE` exist, the file is treated as a site dataset.
If columns like `SAMPLEID` / `SAMPLETYPE` exist, the file is treated as a sample dataset.

## Current behavior
- Initial map view is set to Western Australia
- After upload, the map automatically zooms to the uploaded points
- Site and sample data use different colors
