# %% [markdown]
# # Inventory management project

# %% [markdown]
# ## Data exploration and preparation

# %% [markdown]
# ### Data loading

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# load data

df_infos = pd.read_csv("infos.csv", sep = "|")
df_items = pd.read_csv("items.csv", sep = "|")
df_orders = pd.read_csv("orders.csv.zip", sep = "|", compression="zip")

# %% [markdown]
# ### Train and Test split

# %%
# fix datetime format for transaction time

df_orders['time'] = pd.to_datetime(df_orders['time'])

# %%
len(df_orders["itemID"].unique())

# %%
# Define the split date
split_date = pd.to_datetime("08.06.2018", format="%d.%m.%Y")

df_orders['time'] = pd.to_datetime(df_orders['time'])  # ensure it's datetime
split_date_only = split_date.date()

df_test = df_orders[df_orders['time'].dt.date > split_date_only]
df_train = df_orders[df_orders['time'].dt.date <= split_date_only]

# %%
# we saw that we lose around 1000 items with the train-test split, because they have orders only in the last 3 weeks
# so we check their price x quantity (revenue) - what % it is of the total revenue
# to see if we lose a lot with this cropping and decide how to continue

# %%
df_test["revenue"] = df_test["order"] * df_test["salesPrice"]
df_orders["revenue"] = df_orders["order"] * df_orders["salesPrice"]

# %%
missing_items = pd.DataFrame(df_test[~df_test["itemID"].isin(df_train["itemID"])]["itemID"].unique(), columns = ["itemID"])
missing_items["will_be_lost"] = "yes"

# %%
df_orders = df_orders.merge(missing_items, how = "left", on = "itemID")

# %%
df_orders["revenue"] = df_orders["salesPrice"] * df_orders["order"]

# %%
round((df_orders[df_orders["will_be_lost"] == "yes"]["revenue"].sum()/df_orders["revenue"].sum())*100, 2)

# %%
# they account for 5.09% of the total revenue of the historical 6-month data we have
# it is low enough, we continue like this

# %% [markdown]
# ### df_train preparation

# %%
df_train.info()

# %%
# add one column which is only with the date, no time

# train
df_train['date'] = df_train['time'].dt.date
df_train['date'] = pd.to_datetime(df_train['date'])

# test
df_test['date'] = df_test['time'].dt.date
df_test['date'] = pd.to_datetime(df_test['date'])

# %%
# there seem to be 0 price transactions

len(df_train[df_train["salesPrice"]==0])/len(df_train)*100

# they are 0.02% of all transactions, it is best to delete them instead of thinking how to handle them

df_train = df_train[df_train["salesPrice"]!=0]

# and now for test

df_test = df_test[df_test["salesPrice"]!=0]

# %%
# do all items have an order in the period we have?

round(df_train["itemID"].nunique()/len(df_items)*100, 2)

# 94.05% of all items have an order in the period

# %%
# which transactions are performed on a discounted price?
# we assume the following: a transaction is marked as having a promotion if, for the same item, somewhere else in the table there is
# another transaction performed on a lower price

# Step 1: Get the maximum price per itemID
max_price_per_item = df_train.groupby('itemID')['salesPrice'].transform('max')

# Step 2: Compare each row's price to the max price for that item
df_train['promotion'] = df_train['salesPrice'] < max_price_per_item

# Step 3: Convert boolean to "yes"/"no"
df_train['promotion'] = df_train['promotion'].map({True: 1, False: 0})


# and now for test

# Step 1: Get the maximum price per itemID
max_price_per_item = df_test.groupby('itemID')['salesPrice'].transform('max')

# Step 2: Compare each row's price to the max price for that item
df_test['promotion'] = df_test['salesPrice'] < max_price_per_item

# Step 3: Convert boolean to "yes"/"no"
df_test['promotion'] = df_test['promotion'].map({True: 1, False: 0})

# %%
# this data frame will be aggregared on day level for final use
# so it is best if we continue the transformation in the aggregated version
# but before aggregation, we should check what % of items have been sold on a different price in the same day

price_variations = (
    df_train
    .assign(date=df_train['time'].dt.date)
    .groupby(['itemID', 'date'])['salesPrice']
    .nunique()
    .reset_index(name='unique_price_count')
)


# Filter where price count > 1 (i.e., same item sold at multiple prices)
price_variations[price_variations['unique_price_count'] > 1]

# %%
# % of such cases from all items

((price_variations[price_variations['unique_price_count'] > 1]['unique_price_count'].count())/len(df_items))*100

# %% [markdown]
# ### df_infos preparation

# %% [markdown]
# #### Preparation

# %%
# in df_infos column promotion there are cells with more than one date, separated by a comma
# how many such are there?

(df_infos["promotion"].str.len() > 10).sum()

# 190
# I leave it as text for now, we should handle it later

# %%
df_infos["promotion"][df_infos["promotion"].str.len() > 10]

# %%
# does df_infos, containing the promotions, contain unique item IDs or are they duplicated?
# I expect them to be unique

len(df_infos["itemID"]) == len(df_items)
df_infos["itemID"].value_counts().max() == 1


# %%
df_infos["itemID"].isin(df_items["itemID"]).count() == len(df_infos["itemID"])

# it contains a row for each itemID

# %% [markdown]
# #### Deriving discounts for the simulation period

# %%
df_train["maxPrice"] = df_train.groupby("itemID")["salesPrice"].transform("max")

# and for test

df_test["maxPrice"] = df_test.groupby("itemID")["salesPrice"].transform("max")

# %%
# deriving discounts

df_train["discountAmount"] = round(df_train["maxPrice"] - df_train["salesPrice"], 2)

df_train["discountPerc"] = round(df_train["discountAmount"]/df_train["maxPrice"], 2)

# and for test

df_test["discountAmount"] = round(df_test["maxPrice"] - df_test["salesPrice"], 2)

df_test["discountPerc"] = round(df_test["discountAmount"]/df_test["maxPrice"], 2)

# %%
# max and min % discount

print(max(df_train["discountPerc"]), min(df_train["discountPerc"][df_train["discountPerc"] != 0]))

# %%
# Create the histogram and get the bars
ax = df_train["discountPerc"].plot(kind="hist", bins=10, edgecolor='black')

# Add value labels on top of each bar
for patch in ax.patches:
    height = patch.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', 
                    xy=(patch.get_x() + patch.get_width() / 2, height), 
                    xytext=(0, 5),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.xlabel("Discount Percentage")
plt.ylabel("Frequency")
plt.title("Histogram of Discount Percentage")
plt.tight_layout()
plt.show()

# %%
# Total number of observations
total = len(df_train["discountPerc"].dropna())

# Plot the histogram as density (normalized)
ax = df_train["discountPerc"].plot(kind="hist", bins=10, edgecolor='black', density=False)

# Get the actual bin heights (counts) to calculate percentages
counts, bins, patches = plt.hist(df_train["discountPerc"].dropna(), bins=10, edgecolor='black')

# Annotate bars with percentage labels
for count, patch in zip(counts, patches):
    percentage = 100 * count / total
    if count > 0:
        plt.annotate(f'{percentage:.1f}%', 
                     xy=(patch.get_x() + patch.get_width() / 2, count), 
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom')

plt.xlabel("Discount Percentage")
plt.ylabel("Count")
plt.title("Histogram of Discount Percentage (with % labels)")
plt.tight_layout()
plt.show()

# %%
# looking at the skewed distribution, for getting an approximate discount percentage per item
# it would be better to use the median instead of the mean
# adding column for discounted price to table df_infos = simulation price - median discount for item
# Start with itemID column
df_discount_stats = df_items[["itemID"]].copy()

# Filter out rows where discountAmount is 0
df_nonzero_discounts = df_train[df_train["discountAmount"] != 0].copy()

# Drop duplicates to keep only unique discount percentages per item
unique_discounts = df_nonzero_discounts.drop_duplicates(subset=["itemID", "discountPerc"])

# Now compute the median of these unique values per item
median_discounts = (
    unique_discounts
    .groupby("itemID")["discountPerc"]
    .median()
    .round(2)
    .reset_index()
    .rename(columns={"discountPerc": "medianDiscPerc"})
)

# Merge into df_discount_stats
df_discount_stats = df_discount_stats.merge(median_discounts, on="itemID", how="left")



# %%
# adding column for discounted price to table df_infos = simulation price - median discount for item

df_infos = df_infos.merge(df_discount_stats[['itemID', 'medianDiscPerc']], on='itemID', how='left')

# %%
# adding column for discounted price to table df_infos = simulation price - median discount for item

df_infos["discountedPrice"] = np.where(
    df_infos["promotion"].notna(),
    round(df_infos["simulationPrice"] * (1 - df_infos["medianDiscPerc"]), 2),
    np.nan  # or just leave it to default if you prefer
)


# %%
# unfinished - we have to use some mean based on similar items to derive median discount % for items which will have
# a promotion in the simulation period but have not had a discount in the historical data

# %%
# adding also min price per item in the orders data frame for completion

df_train["minPrice"] = df_train.groupby("itemID")["salesPrice"].transform("min")

# and for test

df_test["minPrice"] = df_test.groupby("itemID")["salesPrice"].transform("min")

# %% [markdown]
# #### Quickly check relation - qty sold and promotion

# %%
# promo tests

# Step 1: Sum quantity per itemID, date, and promotion (daily sales)
daily_qty = (
    df_train
    .groupby(['itemID', 'date', 'promotion'])['order']
    .sum()
    .reset_index()
)

# Step 2: Aggregate by itemID and promotion: 
# total quantity sold (sum of daily sums)
# count of days with sales (number of unique days)
agg = daily_qty.groupby(['itemID', 'promotion']).agg(
    total_qty=('order', 'sum'),
    count_days=('date', 'nunique')
).unstack(fill_value=0)

# Step 3: Build the final DataFrame safely extracting promo/no promo columns
summary = pd.DataFrame({
    'QTY_no_promo': agg['total_qty'].get(0, pd.Series(0)),
    'QTY_promo': agg['total_qty'].get(1, pd.Series(0)),
    'count_days_no_promo': agg['count_days'].get(0, pd.Series(0)),
    'count_days_promo': agg['count_days'].get(1, pd.Series(0))
}).reset_index()

# Step 4: Calculate average quantity per day (handle division by zero)
summary['QTY_no_promo_per_day'] = summary.apply(
    lambda r: r['QTY_no_promo'] / r['count_days_no_promo'] if r['count_days_no_promo'] > 0 else 0,
    axis=1
)
summary['QTY_promo_per_day'] = summary.apply(
    lambda r: r['QTY_promo'] / r['count_days_promo'] if r['count_days_promo'] > 0 else 0,
    axis=1
)


# %%
# promo tests

len(summary[summary["QTY_promo_per_day"] > summary["QTY_no_promo_per_day"]])/len(summary)

# %% [markdown]
# ## Aggregate orders

# %%
# aggregate df_train on a daily basis
# sum of QTY
# average of price (or median?)?
# promotion - if 1 is present, then 1 (had at least 1 promotion in that day)
# median discount %?
# median discount amount?

# to make a desicion wether to use mean of median for price, discount amount, discount perc
# we have to look at the distribution of the prices for some items


# %% [markdown]
# ### Checking price per item distributions

# %%
# adding a column with item_prices_count to df_train

df_train["item_prices_count"] = df_train.groupby("itemID")["salesPrice"].transform(lambda x: x.nunique())

# %%
# getting a random sample where the item has price discount of > 0.79, meaning there might
# be great price variations of the item

df_sample = df_train[["itemID"]][df_train["discountPerc"] > 0.79]

df_sample = df_sample.sample(n=50, random_state=222)

df_sample = df_sample.sort_values(by="itemID", ascending=True)

df_sample = df_sample.merge(df_train, how="left", on="itemID")

# %%
# visualizing 

# Unique items
item_ids = df_sample['itemID'].unique()

# Set up the grid
rows, cols = 10, 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharex=False, sharey=False)
axes = axes.flatten()

# Plot histogram for each item
for i, item_id in enumerate(item_ids):
    ax = axes[i]
    item_prices = df_sample[df_sample['itemID'] == item_id]['salesPrice']

    ax.hist(item_prices, bins=10, color='skyblue', edgecolor='black')
    ax.set_title(f'Item {item_id}', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)

# Hide unused subplots
for j in range(len(item_ids), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
# it would be useful to see another check to make aa decision
# see the top 10 items for which the mean and median are the most different and
# see which one makes more sence for us

# Group by itemID and compute mean and median
price_stats = df_train.groupby('itemID')['salesPrice'].agg(
    mean_price='mean',
    median_price='median'
).reset_index()

# Compute absolute difference
price_stats['abs_diff'] = (price_stats['mean_price'] - price_stats['median_price']).abs()

# Compute absolute percentage difference relative to median
price_stats['abs_perc_diff'] = (price_stats['abs_diff'] / price_stats['median_price']).abs() * 100

# Round numerical columns
price_stats[['mean_price', 'median_price', 'abs_diff', 'abs_perc_diff']] = price_stats[
    ['mean_price', 'median_price', 'abs_diff', 'abs_perc_diff']].round(2)

# Sort by absolute percentage difference descending
price_stats = price_stats.sort_values(by='abs_perc_diff', ascending=False)

# %%
# now look at histograms of top 50 of items with most % difference of mean and median

df_sample = price_stats.sort_values(by='abs_perc_diff', ascending=False).head(50)[["itemID"]]

df_sample = df_sample.sample(n=50, random_state=222)

df_sample = df_sample.sort_values(by="itemID", ascending=True)

df_sample = df_sample.merge(df_train, how="left", on="itemID")

# %%
# visualizing 

# Unique items
item_ids = df_sample['itemID'].unique()

# Set up the grid
rows, cols = 10, 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharex=False, sharey=False)
axes = axes.flatten()

# Plot histogram for each item
for i, item_id in enumerate(item_ids):
    ax = axes[i]
    item_prices = df_sample[df_sample['itemID'] == item_id]['salesPrice']

    ax.hist(item_prices, bins=10, color='skyblue', edgecolor='black')
    ax.set_title(f'Item {item_id}', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)

# Hide unused subplots
for j in range(len(item_ids), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
# it is better to use mean because the extreme values are not a one-case accidental thing

# %%
# aggregate using mean (average), but specifically weighted average, to gain price per each
# weighted average will take into account how accidental the outlier prices were

# %% [markdown]
# ### Aggregate orders on day level

# %%
# in order to get weighted average price per each for the items
# first we need to add a column to df_train
# with order value = qty * price

df_train["orderValue"] = df_train["order"] * df_train["salesPrice"]

# and for test

df_test["orderValue"] = df_test["order"] * df_test["salesPrice"]

# %%
# aggregating on day level

df_train_daily = df_train.groupby(['date', 'itemID']).agg(
    qty_sold=('order', 'sum'),
    sales_value=('orderValue', 'sum'),
    promotion=('promotion', 'max'),  # If any transaction had promotion == 1, result will be 1
    maxItemPrice=('maxPrice', 'max'), # doesn't matter min or max - its the same value for all transactions with the same item
    minItemPrice=('minPrice', 'max') # doesn't matter min or max - its the same value for all transactions with the same item
).reset_index()

# and for test

df_test_daily = df_test.groupby(['date', 'itemID']).agg(
    qty_sold=('order', 'sum'),
    sales_value=('orderValue', 'sum'),
    promotion=('promotion', 'max'),  # If any transaction had promotion == 1, result will be 1
    maxItemPrice=('maxPrice', 'max'), # doesn't matter min or max - its the same value for all transactions with the same item
    minItemPrice=('minPrice', 'max') # doesn't matter min or max - its the same value for all transactions with the same item
).reset_index()

# %%
# deriving price per each for the items, but specific price for that day

df_train_daily["DailyItemQty"] = df_train_daily.groupby(["itemID", "date"])["qty_sold"].transform("sum")
df_train_daily["DailyItemValue"] = df_train_daily.groupby(["itemID", "date"])["sales_value"].transform("sum")

df_train_daily["PricePerEachToday"] = round(df_train_daily["DailyItemValue"] / df_train_daily["DailyItemQty"], 2)

df_train_daily = df_train_daily.drop(['DailyItemQty', 'DailyItemValue'], axis=1)

# and for test

df_test_daily["DailyItemQty"] = df_test_daily.groupby(["itemID", "date"])["qty_sold"].transform("sum")
df_test_daily["DailyItemValue"] = df_test_daily.groupby(["itemID", "date"])["sales_value"].transform("sum")

df_test_daily["PricePerEachToday"] = round(df_test_daily["DailyItemValue"] / df_test_daily["DailyItemQty"], 2)

df_test_daily = df_test_daily.drop(['DailyItemQty', 'DailyItemValue'], axis=1)

# %%
# deriving just average price (for the day and as a whole), no QTY used in order to do weighted average

df_train_daily["Price"] = round(df_train_daily.groupby("itemID")["sales_value"].transform("mean"), 2)

df_train_daily["PriceToday"] = df_train_daily.groupby(["itemID", "date"])["sales_value"].transform("mean")

# and for test

df_test_daily["Price"] = round(df_test_daily.groupby("itemID")["sales_value"].transform("mean"), 2)

df_test_daily["PriceToday"] = df_test_daily.groupby(["itemID", "date"])["sales_value"].transform("mean")

# %%
# add median discout for the items
# we have it currently for each item in df_infos

# Select only itemID and medianDiscount from df_infos and merge on itemID
df_train_daily = df_train_daily.merge(
    df_infos[["itemID", "medianDiscPerc"]],
    how="left",
    on="itemID"
)

# and for test

df_test_daily = df_test_daily.merge(
    df_infos[["itemID", "medianDiscPerc"]],
    how="left",
    on="itemID"
)

# %% [markdown]
# ## Add features

# %% [markdown]
# #### Complete main DF with missing date + item combinations

# %%
# completing our main data frame with all missing day+item combinations
# for them qty_sold = 0, sales_value = 0, promotion = 0, maxItemPrice = maxItemPrice, minItemPrice = minItemPrice
# Price = mean Price for that item, medianDiscPerc = medianDiscPerc

new_df_train = pd.DataFrame(df_train_daily["itemID"].unique(), columns = ["itemID"])

# and for test

new_df_test = pd.DataFrame(df_test_daily["itemID"].unique(), columns = ["itemID"])

# %%
# Create date range
#checkp

date_range = pd.date_range(start='2018-01-01', end='2018-06-08', freq='D')

# Create DataFrame
df_dates_train = pd.DataFrame({'date': date_range})

# and for test

# Create date range
date_range = pd.date_range(start='2018-06-09', end='2018-06-29', freq='D')

# Create DataFrame
df_dates_test = pd.DataFrame({'date': date_range})

# %%
# Add a dummy key to both DataFrames
df_dates_train["key"] = 1
new_df_train["key"] = 1

# Perform cross join
new_df_train = pd.merge(df_dates_train, new_df_train, on="key").drop("key", axis=1)

# Sort by date and then itemID
new_df_train = new_df_train.sort_values(by=["date", "itemID"]).reset_index(drop=True)

# and for test
df_dates_test["key"] = 1
new_df_test["key"] = 1

# Perform cross join
new_df_test = pd.merge(df_dates_test, new_df_test, on="key").drop("key", axis=1)

# Sort by date and then itemID
new_df_test = new_df_test.sort_values(by=["date", "itemID"]).reset_index(drop=True)

# %%
df_train_daily = new_df_train.merge(df_train_daily, how="left", on=["date", "itemID"])

# and for test

df_test_daily = new_df_test.merge(df_test_daily, how="left", on=["date", "itemID"])

# %%
# for them qty_sold = 0, sales_value = 0, promotion = 0

df_train_daily[["qty_sold", "sales_value", "promotion"]] = df_train_daily[["qty_sold", "sales_value", "promotion"]].fillna(0)

# and for test

df_test_daily[["qty_sold", "sales_value", "promotion"]] = df_test_daily[["qty_sold", "sales_value", "promotion"]].fillna(0)

# %%
# maxItemPrice = maxItemPrice, minItemPrice = minItemPrice, Price = mean Price for that item, medianDiscPerc = medianDiscPerc

# Fill missing maxItemPrice with the max per itemID
df_train_daily["maxItemPrice"] = df_train_daily.groupby("itemID")["maxItemPrice"].transform(lambda x: x.fillna(x.max()))

# Fill missing minItemPrice with the max per itemID
df_train_daily["minItemPrice"] = df_train_daily.groupby("itemID")["minItemPrice"].transform(lambda x: x.fillna(x.max()))

# Fill missing Price with the max per itemID
df_train_daily["Price"] = df_train_daily.groupby("itemID")["Price"].transform(lambda x: x.fillna(x.max()))

# Fill missing PricePerEachToday with the mean per itemID
df_train_daily["PricePerEachToday"] = df_train_daily.groupby("itemID")["PricePerEachToday"].transform(
    lambda x: x.fillna(x[df_train_daily.loc[x.index, "qty_sold"] != 0].mean())
).round(2)

# Fill missing PriceToday with the mean per itemID
df_train_daily["PriceToday"] = df_train_daily.groupby("itemID")["PriceToday"].transform(
    lambda x: x.fillna(x[df_train_daily.loc[x.index, "qty_sold"] != 0].mean())
).round(2)

# Fill missing medianDiscPerc with the max per itemID
df_train_daily["medianDiscPerc"] = df_train_daily.groupby("itemID")["medianDiscPerc"].transform(lambda x: x.fillna(x.max()))

# %%
df_train_daily

# %%
# and for test

# maxItemPrice = maxItemPrice, minItemPrice = minItemPrice, Price = mean Price for that item, medianDiscPerc = medianDiscPerc

# Fill missing maxItemPrice with the max per itemID
df_test_daily["maxItemPrice"] = df_test_daily.groupby("itemID")["maxItemPrice"].transform(lambda x: x.fillna(x.max()))

# Fill missing minItemPrice with the max per itemID
df_test_daily["minItemPrice"] = df_test_daily.groupby("itemID")["minItemPrice"].transform(lambda x: x.fillna(x.max()))

# Fill missing Price with the max per itemID
df_test_daily["Price"] = df_test_daily.groupby("itemID")["Price"].transform(lambda x: x.fillna(x.max()))

# Fill missing PricePerEachToday with the mean per itemID
df_test_daily["PricePerEachToday"] = df_test_daily.groupby("itemID")["PricePerEachToday"].transform(
    lambda x: x.fillna(x[df_test_daily.loc[x.index, "qty_sold"] != 0].mean())
)
df_test_daily["PricePerEachToday"] = df_test_daily["PricePerEachToday"].round(2)

# Fill missing PriceToday with the mean per itemID
df_test_daily["PriceToday"] = df_test_daily.groupby("itemID")["PriceToday"].transform(
    lambda x: x.fillna(x[df_test_daily.loc[x.index, "qty_sold"] != 0].mean())
)
df_test_daily["PriceToday"] = df_test_daily["PriceToday"].round(2)

# Fill missing medianDiscPerc with the max per itemID
df_test_daily["medianDiscPerc"] = df_test_daily.groupby("itemID")["medianDiscPerc"].transform(lambda x: x.fillna(x.max()))

# %% [markdown]
# #### Add masterdata

# %%
# include masterdata

df_train_daily = df_train_daily.merge(df_items, how="left", on="itemID")

# and in test

df_test_daily = df_test_daily.merge(df_items, how="left", on="itemID")

# %% [markdown]
# #### Add date features

# %%
# Add date features
df_train_daily["weekDay"] = df_train_daily["date"].dt.weekday + 1
df_train_daily["day"] = df_train_daily["date"].dt.day

# and in test

df_test_daily["weekDay"] = df_test_daily["date"].dt.weekday + 1
df_test_daily["day"] = df_test_daily["date"].dt.day

# %%
def get_week_of_month(date):
    # First day of the month
    first_day = date.replace(day=1)
    
    # Find the day of the week the first day lands on (Monday=0, Sunday=6)
    first_day_weekday = first_day.weekday()
    
    # Calendar row index = (day of month + offset from Monday) // 7 + 1
    return ((date.day + first_day_weekday - 1) // 7) + 1

# Apply the function to create the column
df_train_daily['weekOfMonth'] = df_train_daily['date'].apply(get_week_of_month)

# and in test
df_test_daily['weekOfMonth'] = df_test_daily['date'].apply(get_week_of_month)

# %% [markdown]
# ### FFT

# %%
def get_harmonics(data, num_harmonics=10, return_wave=None):
    all_coefs = np.fft.fft(data)
    coeffs = []
    nh = return_wave + 1 if return_wave is not None else num_harmonics
    for i in range(1, nh + 1):
        coeffs.append(np.zeros(len(all_coefs), dtype=complex))
        coeffs[-1][i] = all_coefs[i]
        coeffs[-1][-i] = all_coefs[-i]

    if return_wave is not None:
        rc = np.zeros(len(all_coefs), dtype=complex) + coeffs[return_wave]
        rc = np.fft.ifft(rc).real
        return rc

    reconstructed_coeffs = np.zeros(len(all_coefs), dtype=complex)
    for i in range(num_harmonics):
        reconstructed_coeffs += coeffs[i]
    reconstructed_signal = np.fft.ifft(reconstructed_coeffs).real
    reconstructed_signal += data.mean()
    return reconstructed_signal

# %%
for i in range(5):
    df_train_daily[f'harmonic_{i}'] = df_train_daily.groupby(by="itemID")["qty_sold"].transform(lambda c: get_harmonics(c, return_wave=i))

# and in test

for i in range(5):
    df_test_daily[f'harmonic_{i}'] = df_test_daily.groupby(by="itemID")["qty_sold"].transform(lambda c: get_harmonics(c, return_wave=i))

# %%
df_train_daily.columns

# %% [markdown]
# ### Cumulative variables

# %%
df_train_daily['cum_sum_order'] = df_train_daily.groupby('itemID')['qty_sold'].cumsum()

# and in test

df_test_daily['cum_sum_order'] = df_test_daily.groupby('itemID')['qty_sold'].cumsum()

# %% [markdown]
# ### Rolling statistics

# %%
# once again, making sure data frame is sorted by date and item ID so the rolling stats are OK

df_train_daily.sort_values(['itemID', 'date'], inplace=True)

# and in test

df_test_daily.sort_values(['itemID', 'date'], inplace=True)

# %%
df_train_daily['rolling_qty_sold_mean'] = (
    df_train_daily
    .groupby('itemID')['qty_sold']
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)

df_train_daily['rolling_qty_sold_std'] = (
    df_train_daily
    .groupby('itemID')['qty_sold']
    .transform(lambda x: x.shift(1).rolling(window=7).std())
)

df_train_daily['rolling_qty_sold_median'] = (df_train_daily
    .groupby('itemID')['qty_sold']
    .transform(lambda x: x.shift(1).rolling(window=7).median())
)

#def median_of_uniques(x):
#    return np.median(np.unique(x))

#df_train_daily = df_train_daily.sort_values(['itemID', 'date'])

#df_train_daily['rolling_qty_sold_median_distincts'] = (
#    df_train_daily
#    .groupby('itemID')['qty_sold']
#    .transform(lambda x: x.shift(1).rolling(window=7).apply(median_of_uniques, raw=True))
#)


# %%
# and in test

df_test_daily['rolling_qty_sold_mean'] = (
    df_test_daily
    .groupby('itemID')['qty_sold']
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)

df_test_daily['rolling_qty_sold_std'] = (
    df_test_daily
    .groupby('itemID')['qty_sold']
    .transform(lambda x: x.shift(1).rolling(window=7).std())
)

df_test_daily['rolling_qty_sold_median'] = (df_test_daily
    .groupby('itemID')['qty_sold']
    .transform(lambda x: x.shift(1).rolling(window=7).median())
)

# %% [markdown]
# ### Add lagged variables

# %% [markdown]
# #### Lagged qty_sold

# %%
# sort by item and date before shift
df_train_daily = df_train_daily.sort_values(by=["itemID", "date"])

# and in test

df_test_daily = df_test_daily.sort_values(by=["itemID", "date"])

# %%
# looking at average daily orders and median daily orders of the items
# to help decide how many lags are appropriate

# define a function to compute the median of unique values
def median_of_unique(x):
    return np.median(np.unique(x))

# compute average daily orders and median of unique daily orders
item_daily_stats = (
    df_train_daily.groupby('itemID')["qty_sold"]
    .agg(
        avg_daily_orders='mean',
        median_daily_orders_unique=lambda x: median_of_unique(x)
    )
    .reset_index()
)

item_daily_stats["avg_daily_orders"] = item_daily_stats["avg_daily_orders"].round(2)
item_daily_stats["median_daily_orders_unique"] = item_daily_stats["median_daily_orders_unique"].round(2)

# %%
# I tested a couple of options
# settled on 1 lag only to be able to drop the rows with missing values  resutlting from the lag
# without losing much data
# anyways, we would expect that the sale from the day directly before will be most significat

# %%
# Create lagged variables for qty_sold (currently, just 1)

lags = [1, 2, 3, 7]
for lag in lags:
    df_train_daily[f"qty_sold_lag{lag}"] = (
        df_train_daily.groupby("itemID")["qty_sold"].shift(lag)
    )

# and in test

lags = [1, 2, 3, 7]
for lag in lags:
    df_test_daily[f"qty_sold_lag{lag}"] = (
        df_test_daily.groupby("itemID")["qty_sold"].shift(lag)
    )

# %%
# Reorder lag columns right after qty_sold
# Get all column names
cols = list(df_train_daily.columns)

# Remove lag columns from current position
lag_cols = [f"qty_sold_lag{lag}" for lag in lags]
for col in lag_cols:
    cols.remove(col)

# Find index of qty_sold
qty_idx = cols.index("qty_sold")

# Insert lag columns in order after qty_sold
for i, col in enumerate(lag_cols):
    cols.insert(qty_idx + 1 + i, col)

# Reorder DataFrame
df_train_daily = df_train_daily[cols]

# %%
# and in test

# Reorder lag columns right after qty_sold
# Get all column names
cols = list(df_test_daily.columns)

# Remove lag columns from current position
lag_cols = [f"qty_sold_lag{lag}" for lag in lags]
for col in lag_cols:
    cols.remove(col)

# Find index of qty_sold
qty_idx = cols.index("qty_sold")

# Insert lag columns in order after qty_sold
for i, col in enumerate(lag_cols):
    cols.insert(qty_idx + 1 + i, col)

# Reorder DataFrame
df_test_daily = df_test_daily[cols]

# %%
# how many values will remain if rows containing NA are dropped

len(df_train_daily.dropna(subset=["qty_sold_lag1", "qty_sold_lag2", "qty_sold_lag3", "qty_sold_lag7"]))

# %% [markdown]
# #### Lagged PricePerEachToday

# %%
# sort by item and date before shift
df_train_daily = df_train_daily.sort_values(by=["itemID", "date"])

# and in test

df_test_daily = df_test_daily.sort_values(by=["itemID", "date"])

# %%
# Create lagged variables for PricePerEachToday

lags = [1, 2, 3, 7]
for lag in lags:
    df_train_daily[f"PricePerEach_lag{lag}"] = (
        df_train_daily.groupby("itemID")["PricePerEachToday"].shift(lag)
    )

# and in test

lags = [1, 2, 3, 7]
for lag in lags:
    df_test_daily[f"PricePerEach_lag{lag}"] = (
        df_test_daily.groupby("itemID")["PricePerEachToday"].shift(lag)
    )

# %%
# remove unlagged column PricePerEachToday

df_train_daily = df_train_daily.drop(columns=["PricePerEachToday"])

# and in test

df_test_daily = df_test_daily.drop(columns=["PricePerEachToday"])

# %% [markdown]
# #### Lagged sales_value

# %%
# sort by item and date before shift
df_train_daily = df_train_daily.sort_values(by=["itemID", "date"])

# and in test

df_test_daily = df_test_daily.sort_values(by=["itemID", "date"])

# %%
# Create lagged variables for PricePerEachToday

lags = [1, 2, 3, 7]
for lag in lags:
    df_train_daily[f"sales_value_lag{lag}"] = (
        df_train_daily.groupby("itemID")["sales_value"].shift(lag)
    )

# and in test

lags = [1, 2, 3, 7]
for lag in lags:
    df_test_daily[f"sales_value_lag{lag}"] = (
        df_test_daily.groupby("itemID")["sales_value"].shift(lag)
    )

# %%
# remove unlagged column PricePerEachToday

df_train_daily = df_train_daily.drop(columns=["sales_value"])

# and in test

df_test_daily = df_test_daily.drop(columns=["sales_value"])

# %%
# how many values will remain if rows containing NA are dropped

len(df_train_daily.dropna(subset=["qty_sold_lag1", "qty_sold_lag2", "qty_sold_lag3", "qty_sold_lag7"]))

# %%
# what % will remain if rows containing NA are dropped

round((len(df_train_daily.dropna(subset=["qty_sold_lag1", "qty_sold_lag2", "qty_sold_lag3", "qty_sold_lag7"]))/len(df_train_daily))*100, 2)

# %%
# dropping rows containing NA

df_train_daily.dropna(subset=["qty_sold_lag1", "qty_sold_lag2", "qty_sold_lag3", "qty_sold_lag7"], inplace=True)

# and in test

df_test_daily.dropna(subset=["qty_sold_lag1", "qty_sold_lag2", "qty_sold_lag3", "qty_sold_lag7"], inplace=True)

# %% [markdown]
# ### Add 2 noise columns for significance testing

# %%
# think about whether we want to put some borders

# %%
np.random.seed(22)  # for reproducibility

df_train_daily["random_noise1"] = np.random.normal(0, 1, len(df_train_daily))

# and in test

np.random.seed(23)  # for reproducibility

df_test_daily["random_noise1"] = np.random.normal(0, 1, len(df_test_daily))

# %%
np.random.seed(66)  # for reproducibility

df_train_daily["random_noise2"] = np.random.normal(0, 1, len(df_train_daily))

# and in test

np.random.seed(67)  # for reproducibility

df_test_daily["random_noise2"] = np.random.normal(0, 1, len(df_test_daily))

# %%
# some more rounding

df_train_daily["random_noise1"] = df_train_daily["random_noise1"].round(2)
df_train_daily["random_noise2"] = df_train_daily["random_noise2"].round(2)

# and in test

df_test_daily["random_noise1"] = df_test_daily["random_noise1"].round(2)
df_test_daily["random_noise2"] = df_test_daily["random_noise2"].round(2)

# %% [markdown]
# ## Exporting results

# %%
df_train_daily.to_csv("train.csv", index=False)
df_test_daily.to_csv("test.csv", index=False)

# %%
df_train_daily.columns

# %%



