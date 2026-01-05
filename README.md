# BitCraft Market Analyzer

A desktop GUI application for **BitCraft Online** that analyzes live market data using the **Bitjita API** to identify **high-volume, high-value trading opportunities** across all regions.

The tool is designed to help players quickly find items with strong market activity by combining **volume**, **price**, and **outlier-filtered averages** into a single ranking score.

> This project is fan-made and not affiliated with BitCraft Online or Bitjita.

---

##  Features

### Market Analysis
- Pulls **live market data** from the Bitjita API
- Supports **all BitCraft regions (R0–R8)** or global view
- Calculates a **Value Score** = `Average Price × Total Volume`
- Filters extreme price outliers for more realistic averages
- Supports **Sell Orders**, **Buy Orders**, or **Both**

### GUI Experience
- Fast, responsive **PySide6 desktop GUI**
- Background loading using worker threads (UI never freezes)
- Real-time **progress bar & status updates**
- Sortable, readable table layout
- **Expandable rows** to inspect individual orders:
  - Claim name
  - Owner
  - Price
  - Quantity
  - Total order value

### Performance & Quality
- Built-in **5-minute caching** per region to avoid API spam
- Manual cache clear & forced rescan
- Item name **autocomplete search**
- Configurable minimum volume & result limits
---

## 📦 Download (No Python Required)

If you just want to **use the app**, download the Windows executable:

👉 Go to **Releases** and download  
**`BitcraftMarketAnalyzer.exe`**

No Python installation required.

> ⚠️ Windows may show a SmartScreen warning because the app is unsigned.  
> Click **More info → Run anyway**.

---

## 🛠️ How It Works (High Level)

1. Fetches market listings from Bitjita
2. Retrieves per-item buy & sell orders
3. Removes extreme price outliers using percentile filtering
4. Aggregates data by item:
   - Total volume
   - Average price
   - Buy/Sell order counts
5. Ranks items by **Value Score**
6. Displays results in an expandable table

---

## ⚙️ Configuration

Key settings are defined in `config.py`:

```python
DEFAULT_MIN_VOLUME = 100
DEFAULT_MAX_RESULTS = 100

OUTLIER_PERCENTILE_LOW = 10
OUTLIER_PERCENTILE_HIGH = 75
```

---

Outliers are removed using percentile filtering to avoid price manipulation

Cache duration: 5 minutes per region

API timeout: 10 seconds

## For Developers

### Requirements

- Python 3.11+

- Windows (Linux/macOS may work but are untested)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run From Source
```bash 
python main.py
```

### Build the EXE (PyInstaller)
This project includes a ready-to-use spec file.

```bash
pyinstaller main.spec
```
Output:
```bash
dist/
└── BitcraftMarketAnalyzer.exe
```

## ❗ Known Limitations

- API speed depends on Bitjita availability
- Very large scans may take several seconds/minutes (progress shown)
- Windows only (EXE build)

## Disclaimer

This project is:
- Fan-made
- Provided as-is
- Not affiliated with BitCraft Online or Bitjita
- All game data belongs to its respective owners.