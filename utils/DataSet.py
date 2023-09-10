import pandas as pd
import numpy as np
import tqdm
from scipy import stats

EPS = np.finfo(float).eps


class Data():

    def __init__(self,
                 year: int,
                 version: str = "HS92",
                 data_path: str = "../data/economic_complexity/",
                 country_dict: dict = None):

        self.gdp_dict = None
        self.country_dict = country_dict
        self.year = year
        self.version = version
        self.data_path = data_path

        self.df = pd.read_csv(data_path + f"BACI_{version}_V202301/BACI_{version}_Y{year}_V202301.csv",
                              dtype={'k': str})
        # adding zero to string of length 5 in the k column in self.df
        self.df.loc[self.df["k"].str.len() == 5, "k"] = "0" + self.df["k"]

        # compute adjacency matrix
        self.adjacency_matrix = get_adjacency_matrix(self.df)

    def get_entropy(self, iteration: int = 100) -> tuple[dict, dict]:
        x = np.array(self.adjacency_matrix)  # from pandas dataframe to np.array matrix

        # to avoid computational problems we replace np.nan values with eps, in fact -x log(x) = 0 for x tend to zero
        x = np.nan_to_num(x, nan=EPS)

        # initiating entropies
        H_c = stats.entropy(x, axis=1)
        H_p = stats.entropy(x, axis=0)
        country_size = x.shape[0]  # number of countries
        product_size = x.shape[1]  # number of products

        # entropy iterative scheme
        for _ in range(iteration):
            f_weight = np.log(country_size) - H_p
            g_weight = np.log(product_size) - H_c
            x1 = x * f_weight
            x2 = (x.T * g_weight).T
            xi = x1 / x1.sum(axis=1, keepdims=True)
            zeta = x2 / x2.sum(axis=0, keepdims=True)
            H_c = stats.entropy(xi, axis=1)
            H_p = stats.entropy(zeta, axis=0)

        # transform the keys into iso3
        H_c_keys = [self.country_dict.get(key) for key in self.adjacency_matrix.index]
        H_p_keys = self.adjacency_matrix.columns

        # create two dictionaries with the entropy values
        H_c_dict = dict(zip(H_c_keys, H_c))
        H_p_dict = dict(zip(H_p_keys, H_p))
        return H_c_dict, H_p_dict

    def get_gdp(self):
        if self.gdp_dict is None:
            gdp_df = pd.read_csv(self.data_path + "API_NY/API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_5734697.csv", skiprows=4)
            gdp_df = gdp_df[['Country Code', str(self.year)]].set_index('Country Code')
            gdp_df[str(self.year)] = np.log(gdp_df[str(self.year)])
            self.gdp_dict = gdp_df.to_dict()[str(self.year)]
        return self.gdp_dict

    def get_product_count(self):
        # count the number of products exported by each country
        keys = [self.country_dict.get(key) for key in self.adjacency_matrix.index]
        my_dict = self.adjacency_matrix.count(axis=1).to_dict()
        # change the keys
        my_dict = dict(zip(keys, my_dict.values()))
        return my_dict

    def upload_country_df(self):
        country_df = pd.read_csv(self.data_path + f"BACI_{self.version}_V202301/country_codes_V202301.csv")
        # transform product code into iso3
        country_dict = pd.Series(
            data=country_df["iso_3digit_alpha"].values,
            index=country_df["country_code"].values).to_dict()
        self.country_dict = country_dict


"""
    compute adjacency matrix given a dataframe, returns a dataframe with index country code and columns product code
"""


def get_adjacency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(['k', 'i'])
    adjacency_matrix = grouped[['v']].sum().reset_index()
    adjacency_matrix = adjacency_matrix.pivot(index='i', columns='k', values='v').reset_index()
    adjacency_matrix = adjacency_matrix.set_index('i')
    adjacency_matrix.index = adjacency_matrix.index.astype(int)
    adjacency_matrix.columns = adjacency_matrix.columns.astype(str)
    return adjacency_matrix * 1000  # to get values in USD instead of thousands of USD


def get_iso3(country_df: pd.DataFrame) -> dict:
    country_series = pd.Series(
        data=country_df["iso_3digit_alpha"].values,
        index=country_df["country_code"].values
    )
    return country_series.to_dict()


def dynamic_data(years: list[int], version: str = "HS92",
                 data_path: str = "../data/economic_complexity/"):
    # upload the data
    print("Uploading data and computing adjacency...")
    # upload country df
    country_df = pd.read_csv(data_path + f"BACI_{version}_V202301/country_codes_V202301.csv")
    # transform product code into iso3
    country_dict = pd.Series(
        data=country_df["iso_3digit_alpha"].values,
        index=country_df["country_code"].values).to_dict()
    df = pd.DataFrame()
    for year in tqdm.tqdm(years):
        data = Data(year=year, version=version, data_path=data_path, country_dict=country_dict)
        # compute entropy
        entropy_country, entropy_product = data.get_entropy()
        # compute gdp
        gdp_country = data.get_gdp()
        # create dataframe
        df_year = pd.DataFrame(index=entropy_country.keys())
        df_year['country'] = df_year.index
        df_year['year'] = [year] * len(df_year)
        df_year['entropy'] = [entropy_country[index] for index in df_year.index]
        df_year['log gdpPPP'] = [gdp_country.get(index) for index in df_year.index]
        # concatenate
        df = pd.concat([df, df_year.reset_index(drop=True)])
    return df
# %%
