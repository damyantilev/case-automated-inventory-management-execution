# Project Brief: Demand Forecasting for Optimized Inventory Planning

## Objective

Develop a machine learning model to forecast the 14-day demand for each product listed in `items.csv`, starting from **June 30, 2018**, using real anonymized retail data. The goal is to **maximize the monetary value** for a retailer by accurately predicting sales while avoiding overstocking.

## Data Sources

You are provided with three structured CSV files:

1. **`items.csv`** – Master data with descriptive features for each item (categorical/numerical).  
2. **`orders.csv`** – Detailed historical transactions over 6 months, including timestamps (not pre-aggregated).  
3. **`infos.csv`** – Item prices and promotion dates for the simulation (forecast) period.

> **Note:** Column details and value ranges are available in the accompanying `features.pdf` document. Fields are separated by `|`.

## Task Requirements

- **Aggregate** historical demand per item (e.g., daily or weekly) from `orders.csv`.  
- **Create features** from all three files (`items.csv`, `orders.csv`, `infos.csv`).  
- **Build** a machine learning model to predict total demand for each item for the 14-day period starting June 30, 2018.  
- **Incorporate promotions** (explicitly flagged in `infos.csv`) into your model.  
- **Assume prices are fixed** during the forecast period (no need to model price effects).

## Output File Format

Submit predictions in a CSV file with the following structure:

```text
itemID|demandPrediction
995539|34
1000002|42
995554|10
````

* `itemID`: string – unique identifier (as in `items.csv`)
* `demandPrediction`: integer – total forecasted demand for the 14 days
* **Separator**: `|`
* **Filename**: `<TeamName>.csv` (e.g., `TU_Chemnitz_1.csv`)

## Evaluation Metric

Submissions will be evaluated based on their **monetary value** to the retailer:

* If **forecast ≤ actual demand**:

  ```
  Value = price × forecasted_demand
  ```
* If **forecast > actual demand**:

  ```
  Value = price × actual_demand 
        – 0.6 × price × (forecast – actual)
  ```

The goal is to maximize total monetary value over all items by balancing revenue and overstock penalties.

## Additional Requirement

Propose a model or approach in which the retailer can explicitly define the following constraints:

* **Maximum acceptable overstock** (waste).
* **Maximum acceptable lost sales** (missed revenue opportunities).
* **Maximum acceptable locked capital in inventory**, assuming a 30% markup over the product’s purchase cost.

## Exploratory data analysis

1. We don't have information on which transactions have happened under promotion.
 - We identify a promotional transaction by comparing the price of the product in a transaction with the maximum price of this product for all transactions prior to the current one.
 - If the price of the product in a transaction is less than the cummulative maximum we consider it to be a transaction in a promotion.

2. There are products that sell for multiple prices in a day.
 - When we aggregate the data we calculate the weighted averge price (WAP) using the order quantity sold, i.e. WAP = mean(salePrice / order)

3. Zero prices
 - There are transactions that have happened at 0 salePrice. We remove these from the dataset as they represent ~0.02 (TODO: Factcheck it)

4. Sparse data on number of days a product was sold in
 - Most of the products are sold in very small number of days.
 - Kalman filtering - filter values that are with low frequency

## Feature engineering

1. Discount
2. Datetime
3. Customer rating
4. Lagging
5. Rolling window
6. Filtering
 - Kalman

## Modeling
1. SARIMAX
2. FFT


Company plans to restock its inventory every other week and only keep in stock the items that it has actually sold during that period.

Products that are promoted during the simulation period will be earmarked (personal note: that is why the dates in the infos table with the promotions are all in the period which we are simulating).

The model does not need to be able to respond to price changes during the simulation period.

The info file (“infos.csv”) is for the simulation period.
