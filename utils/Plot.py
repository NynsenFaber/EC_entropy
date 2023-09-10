import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class VectorField:

    def __init__(self, df, grid,
                 xlabel="entropy", ylabel="log gdpPPP", index="country", year_step=5):
        self.df = df
        self.grid = grid
        self.xlabel = xlabel
        self.ylabel = ylabel

        ####### SETTING THE GRID #######
        M_H = max(df[xlabel])
        m_H = min(df[xlabel])
        x = np.linspace(m_H, M_H, grid)
        M_e = np.nanmax(df[ylabel])
        m_e = np.nanmin(df[ylabel])
        y = np.linspace(m_e, M_e, grid)
        # intermediate point
        X = np.zeros(grid - 1)
        Y = np.zeros(grid - 1)
        for i in range(1, len(x)):
            X[i - 1] = x[i - 1] + (x[i] - x[i - 1]) / 2
        for i in range(1, len(y)):
            Y[i - 1] = y[i - 1] + (y[i] - y[i - 1]) / 2

        xx, yy = np.meshgrid(X, Y)
        # velocity fields
        u = np.zeros(xx.shape)
        v = np.zeros(yy.shape)
        # color field
        C = np.zeros(yy.shape)
        # variance fields of the components
        variance_y = np.zeros(yy.shape)
        variance_x = np.zeros(yy.shape)

        # iterate over the grid
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                area = (x[i] - x[i - 1]) * (y[j] - y[j - 1])  # area of the box
                delta_x_list = []
                delta_y_list = []
                # get all the data into the box
                data = df[(df[ylabel] <= y[j]) & (df[ylabel] > y[j - 1]) &
                          (df[xlabel] <= x[i]) & (df[xlabel] > x[i - 1])]
                # continue if empty grid
                if data.empty:
                    continue
                # list of countries into the box
                country_list = list(set(data[index]))
                for country in country_list:
                    # we assure time causality in the index of the country dataframe
                    df_country = data[data[index] == country].sort_values(by=["year"]).reset_index(drop=True)
                    years = df_country["year"].values
                    # dataframe of the country evolution
                    df_ev = df[(df[index] == country) & (df["year"].isin(years + year_step))] \
                        .sort_values(by=["year"]).reset_index(drop=True)
                    # to take only the years with a known evolution
                    years = np.intersect1d(years, df_ev["year"].values - year_step)
                    df_country = df_country[df_country["year"].isin(years)]
                    df_ev = df_ev[df_ev["year"].isin(years + year_step)]
                    # new dataframe of the displacements
                    DF_list = []
                    for year in years:
                        a1 = df_country[df_country["year"] == year][xlabel].values[0]
                        a2 = df_country[df_country["year"] == year][ylabel].values[0]
                        b1 = df_ev[df_ev["year"] == year + year_step][xlabel].values[0]
                        b2 = df_ev[df_ev["year"] == year + year_step][ylabel].values[0]
                        df_to_concat = pd.DataFrame.from_records([{
                            "year": year,
                            "{}".format(xlabel): b1 - a1,
                            "{}".format(ylabel): b2 - a2
                        }])
                        DF_list.append(df_to_concat)

                    if not DF_list:
                        continue
                    else:
                        DF = pd.concat(DF_list)

                    # COMPUTING VARIATION
                    delta_x = DF[xlabel].values
                    delta_y = DF[ylabel].values
                    if len(delta_x_list) == 0:
                        delta_x_list = delta_x
                        delta_y_list = delta_y
                    else:
                        # delta_x(y)_list is already filled
                        delta_x_list = np.concatenate((delta_x_list, delta_x))
                        delta_y_list = np.concatenate((delta_y_list, delta_y))
                if len(delta_x_list) > 1:
                    # we opt for the median as a truly central tendency indicator
                    u[j - 1, i - 1] = np.nanmedian(delta_x_list)
                    v[j - 1, i - 1] = np.nanmedian(delta_y_list)

                    # computing variance matrix
                    cov_matrix = np.column_stack((delta_x_list, delta_y_list)).astype(float)
                    # return_cov_matrix.append(cov_matrix)
                    if len(cov_matrix) >= 2:
                        cov = np.cov(np.transpose(cov_matrix))
                        tot_var = np.trace(cov)
                        C[j - 1, i - 1] = tot_var / area
                        variance_y[j - 1, i - 1] = cov[1, 1] / area
                        variance_x[j - 1, i - 1] = cov[0, 0] / area

        self.xx = xx
        self.yy = yy
        self.u = u
        self.v = v
        self.C = C
        self.variance_y = variance_y
        self.variance_x = variance_x

    def flow_plot(self, index="country", save=None):
        plt.figure(figsize=[12, 10])
        country_list = list(set(self.df[index]))
        for i in country_list:
            data = self.df[self.df[index] == i]
            plt.plot(data[self.xlabel], data[self.ylabel], zorder=0, c='0.70', alpha=0.5)
        # normalization of the colors
        C = self.C
        C_0 = C.flatten()
        C_0 = C_0[C_0 != 0]
        threshold = np.quantile(C_0, 0.95)
        C_plot = np.where(C > threshold, threshold, C)
        qq = plt.quiver(self.xx, self.yy, self.u,
                        self.v, C_plot, cmap=plt.cm.plasma, zorder=1, width=0.005)
        cbar = plt.colorbar(qq)
        cbar.set_label("$\sigma^2$", rotation=270, labelpad=+30, fontsize=20)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(15)
        plt.xlim(self.xx.min() - 0.2, self.xx.max() + 0.2)
        # plt.ylim(2.7, 5.3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(f"{self.xlabel}", fontsize=20)
        plt.ylabel(f"{self.ylabel}", fontsize=20)
        if save is not None:
            plt.savefig("Plots/{}".format(save))
        plt.show()

    def trajectory_plot(self, index="country", save=None):
        plt.figure(figsize=[12, 10])
        country_list = list(set(self.df[index]))
        for i in country_list:
            data = self.df[self.df[index] == i]
            plt.plot(data[self.xlabel], data[self.ylabel], zorder=0, alpha=0.5)
            plt.scatter(data[self.xlabel].values[0], data[self.ylabel].values[0], s=5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(f"{self.xlabel}", fontsize=20)
        plt.ylabel(f"log {self.ylabel}", fontsize=20)
        if save is not None:
            plt.savefig("Plots/{}".format(save))
        plt.show()