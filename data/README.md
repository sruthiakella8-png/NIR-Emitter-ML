# Data

The raw data are **not included** in this repository. They belong to the original authors:

> Joung, J. F. *et al.* Experimental database of optical properties of organic compounds. *Scientific Data* **7**, 295 (2020).

1. Download the database file (`DB for chromophore_Sci_Data_rev02.csv`) from the data repository linked in the paper.
2. Place it in `data/raw/`.
3. Run `notebooks/01_data_cleaning.ipynb` to produce `data/processed/my_data_clean.csv` (17,850 chromophore–solvent pairs with scaffold-split labels).
