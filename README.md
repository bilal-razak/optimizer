# Optimizer Dashboard

A web-based dashboard for trading strategy parameter optimization using clustering analysis. This tool helps identify optimal parameter combinations from backtesting results through PCA dimensionality reduction, K-Means clustering, and HDBSCAN density-based clustering.

## Features

- **Step-by-Step Workflow**: Guided 7-step process for parameter optimization
- **Interactive Visualizations**: Plotly-based charts with hover data and full interactivity
- **Multiple Clustering Methods**: K-Means for initial grouping, HDBSCAN for density-based refinement
- **Heatmap Analysis**: Parameter landscape visualization with cluster highlighting
- **Dark/Light Theme**: Toggle between themes for comfortable viewing
- **Core Points Analysis**: Identify high-confidence cluster members

## Quick Start

```bash
# Clone and enter the project directory
cd optimizer_dashboard

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
uvicorn src.api.main:app --reload
```

Open http://localhost:8000 in your browser.

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

1. Create a virtual environment:
```bash
cd optimizer_dashboard
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
# With virtual environment activated:
uvicorn src.api.main:app --reload

# Or without activating (using venv python directly):
.venv/bin/uvicorn src.api.main:app --reload
```

The dashboard will be available at `http://localhost:8000`

### Alternative: Run with Python

```bash
.venv/bin/python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Workflow Steps

#### Step 1: Load Data
- Browse and select your backtesting results CSV file
- Select 2-4 strategy parameters to optimize (e.g., Timeframe, Multiplier, Lookback)
- View data preview and column information

#### Step 2: Heatmaps & Shortlisting
- **Optional Shortlisting**: Filter variants based on metric conditions (e.g., Sharpe Ratio > 1.0)
- **Heatmap Generation**: Visualize parameter performance landscape
  - Select X and Y axis parameters
  - Optionally select a constant parameter for multiple heatmaps
  - Navigate through heatmaps by metric and constant value

#### Step 3: PCA (Principal Component Analysis)
- Reduces dimensionality of multi-metric data
- View explained variance chart
- Interactive scatter plot of PC1 vs PC2

#### Step 4: K-Means Clustering
- Groups variants into K clusters based on metric similarity
- Auto-calculates optimal K or specify manually
- View cluster statistics for different metrics
- Best cluster variants are passed to HDBSCAN

#### Step 5: HDBSCAN Grid Search
- Tests multiple HDBSCAN configurations (min_cluster_size, min_samples)
- Grid visualization of clustering results
- Compare all points vs core points only
- Select optimal configuration for final clustering

#### Step 6: Final HDBSCAN
- Runs HDBSCAN with selected configuration
- Toggle between "All Points" and "Core Points Only" views
- View cluster statistics and distribution

#### Step 7: Best Clusters
- Displays top N clusters ranked by selected metric
- **Heatmaps with Highlighting**: Full parameter landscape with cluster variants highlighted
- **View Toggle**: Switch between "All Variants" and "Core Only" highlighting
- **Filters**: Filter by metric and constant parameter value
- **Cluster Data Table**: View all variants in each cluster

## Input Data Format

The CSV file should contain:
- **Strategy Parameters**: Columns representing tunable parameters (e.g., `TF`, `Multi`, `LB`)
- **Performance Metrics**: Columns like:
  - `sharpe_ratio`
  - `sortino_ratio`
  - `profit_factor`
  - `total_pnl`
  - `win_ratio`
  - `max_draw_down`

Example CSV structure:
```
TF,Multi,LB,sharpe_ratio,sortino_ratio,profit_factor,total_pnl,win_ratio
1,2.0,10,1.5,2.1,1.8,5000,0.55
1,2.0,15,1.3,1.9,1.6,4200,0.52
...
```

## Project Structure

```
optimizer_dashboard/
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css      # Dashboard styling (light/dark themes)
│   │   └── js/
│   │       └── app.js          # Frontend JavaScript application
│   └── templates/
│       └── index.html          # Main HTML template
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI application entry point
│   │   └── routes/
│   │       ├── optimization.py # Single-run optimization endpoint
│   │       └── steps.py        # Step-by-step workflow endpoints
│   ├── models/
│   │   └── optimization.py     # Pydantic request/response models
│   └── optimization/
│       ├── clustering.py       # K-Means and HDBSCAN implementations
│       ├── feature_engineering.py  # PCA and data preprocessing
│       ├── pipeline.py         # Full optimization pipeline
│       └── visualization.py    # Plotly chart generation
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/optimization/browse` | GET | File browser for CSV selection |
| `/optimization/columns` | GET | Get column names from CSV |
| `/steps/load-data` | POST | Step 1: Load and validate data |
| `/steps/shortlist` | POST | Step 2a: Apply shortlisting conditions |
| `/steps/heatmap` | POST | Step 2b: Generate heatmaps |
| `/steps/pca` | POST | Step 3: Run PCA |
| `/steps/kmeans` | POST | Step 4: Run K-Means clustering |
| `/steps/hdbscan-grid` | POST | Step 5: HDBSCAN grid search |
| `/steps/hdbscan-final` | POST | Step 6: Final HDBSCAN with selected config |
| `/steps/best-clusters` | POST | Step 7: Get best clusters and heatmaps |

## Configuration Options

### HDBSCAN Parameters
- **min_cluster_size**: Minimum number of points to form a cluster (default: 5)
- **min_samples**: Minimum samples in neighborhood for core points (default: 3)
- **threshold_cluster_prob**: Probability threshold for core point classification (default: 0.9)

### Ranking Metrics
- `sharpe_ratio` (default)
- `sortino_ratio`
- `profit_factor`

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: PCA and K-Means
- **HDBSCAN**: Density-based clustering
- **Plotly**: Interactive visualizations
- **Jinja2**: HTML templating
