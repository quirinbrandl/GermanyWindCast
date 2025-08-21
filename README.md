# GermanyWindCast

The code for my bachelor thesis: Spatiotemporal Study of
Deep-Learning-Driven Short-Term Wind
Speed Forecasting

## Abstract of the Thesis:

This thesis investigates deep learning for short-term wind speed forecasting through
a systematic spatiotemporal study. In addition to the empirical work, it provides
an in-depth theoretical outline of deep learning methods for regression tasks, such
as wind speed forecasting. We evaluate four architectures: Multilayer perceptron
(MLP), long short-term memory network (LSTM), graph convolutional network
(GCN), and a combination of GCN and LSTM, across forecast horizons of 1, 4,
and 8 hours. The analysis examines how model performance depends on temporal
parameters (input resolution and look-back window) and spatial context (number
of nearby stations). Results show that no architecture consistently dominates:
The GCN-LSTM performs best at 1-hour forecasts, while the simpler MLP excels
at longer horizons. Spatial context proves most influential, with performance
improving steadily up to the maximum of 17 stations. Temporal effects are
secondary but systematic, with finer resolutions generally advantageous and optimal
look-back windows increasing at coarser inputs. Feature selection experiments
indicate that harmonic time encodings improve accuracy at 4- and 8-hour horizons,
while additional meteorological variables beyond wind speed and direction offer
little benefit. Finally, comparison with ICON numerical weather prediction (NWP)
models demonstrates that deep learning achieves especially competitive performance
at 4- and 8-hour horizons, complementing traditional approaches.

## Project Structure

```
GermanyWindCast/
├── src/                    # Main source code
│   ├── models/            # The models employed in the thesis (see Sections 3.1 Forecasting Task, 3.2 Dataset for in thesis for more details)
│   ├── data/              # Data loading and processing (see Section 3.3 Model Architectures in thesis for more details)
│   ├── utils/             # Utillities used in whole project
│   ├── experiments/       # Scripts to automate the feature and spatiotemporal experiments (see Chapter 4 Results in thesis for more details)
│   └── train.py           # Main training script
├── notebooks/             # Jupyter notebooks for analysis
│   ├── exploratory_data_analysis.ipynb
│   ├── temporal_analysis.ipynb
│   ├── spatial_analysis.ipynb
│   └── model_evaluation.ipynb
├── config/               # Configuration files for your next run
│   ├── general_config.yaml
│   └── hyperparameters.yaml
├── data/                 # Data storage
│   ├── raw/              # Raw data files obtained from DWD
│   ├── processed/        # Preprocessed datasets (see preprocessing.py)
│   ├── nwp/              # Forecasts of ICON models obtained via OpenMeteo API
│   └── results/          # Caching of model results (from wandb)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd GermanyWindCast
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## License

This project is licensed under the MIT License.

## Author

**Quirin Brandl**
