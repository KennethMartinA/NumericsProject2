import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import pickle
import argparse

class SpatialInterpolator:
    """Instances of this class will store a collection of Gaussian Process Regressors (from sklearn)
    for the task of spatiotemporal interpolation. The class expects data containing locations (x,y,time)
    along with at least one columns containing some samples of the function f(x,y,t) that the function 
    the GPRs will interpolate.

    Attributes:
        locIndices (np.array) : see below
        valIndices (np.array) : see below
        gprs {string : GaussianProcessRegressor} : a dictionary, where keys are strings corresponding to
        column names in valIndices, and values are (fitted) GPRs
        df (pd.DataFrame) : the initial data that the GPRs are trained on. Most probably it should 
        be garbage collected after fitting is done! //todo
        self.min, self.max (pd.Series) : this contains the respective minimum and maximum of each column
        of the original dataframe used for fitting. It is an array of floats accessible by column names.
        Important for rescale & descale. 
    """
    def __init__(self, raw_df : pd.DataFrame, locIndices : np.array, valIndices : np.array):
        """_summary_

        Args:
            filepath (str): _description_
            locIndices (np.array): array of column names that contain spatiotemporal location data, 
            in the order (x,y,t). (x,y) are expected to be numbers,
            valIndices (np.array): array of integer indices representing the file columns which
            contain the functional data to interpolate. Each additional value columns requires another
            gaussian process regressor 
        """
        self.locIndices = locIndices
        #columns representing samples of the function f(x,y,t) to fit a GP to
        self.valIndices = valIndices
        self.gprs = {}
        
        self.df = raw_df
        #these need to be stored for subsequent descaling/rescaling
        self.min, self.max = self.df.min(), self.df.max()
        #normalize data
        self.df = (self.df - self.min)/(self.max - self.min)
    
    def fit(self, validate: bool = False , 
            kernel = None):
        """The actual fitting function. Everything is done by sklearn, we only make
        sure that there is a separate GPR fitted for each function we want to approximate.
        Notably, there *are* ways to interpolate for multiple functions, but sklearn cannot
        do that.

        Args:
            validate (bool, optional): if true, will do a validate/train split and 
            report final MSE. Not implemented lol. Defaults to False.
            kernel (GaussianProcessRegressor, optional): if desired, will use a kernel different
            from the kernel used in getGPR. Defaults to None.
        """
        X = self.df[self.locIndices]
        for valIndex in self.valIndices:
            Y = self.df[valIndex]
            if kernel == None:
                gpr = SpatialInterpolator.getGPR()
            else:
                gpr = GaussianProcessRegressor(kernel = kernel)
            gpr = gpr.fit(X,Y)
            self.gprs[valIndex] = gpr

    def interpolate(self, nodes : pd.DataFrame, timeIndices : list[int], 
                    filepath : str = None) -> pd.DataFrame:
        """_summary_

        Args:
            nodes (pd.DataFrame): a list of spatial (x,y) locations
            timeIndices (List[int]): an int array corresponding to tempora locations where to predict.
            Method would work fine with floats too.
            filepath (str, optional): supposed to allow for saving the df,
            but not implemented. Defaults to None.

        Returns:
            pd.DataFrame: a df which matches the structure of the original df used for fitting,
            where it contains columns with predicted functional values,except the time
            index *only* ranges over the array timeIndices.
        """
        #we need to construct a new df whose columsn match the data df used for fitting
        nodes_df = pd.DataFrame(columns = self.locIndices[:-1])
        nodes_df.loc[:,self.locIndices[0]] = nodes[0]
        nodes_df.loc[:,self.locIndices[1]] = nodes[1]
        #amend the time column, we add it in a bit
        total_df = pd.DataFrame(columns = self.locIndices[:-1].append("times"))

        #accumulate the df by concatenating (x,y,t_i) for each t_i in timeIndices
        #//todo change this, cause pandas doesnt like it
        for time in timeIndices:
                time_df = nodes_df.copy()
                time_df[self.locIndices[-1]] = time
                total_df = pd.concat([total_df, time_df])
        total_df = self.rescale(total_df)
        for valIndex in self.valIndices:
                    total_df[valIndex] = self.gprs[valIndex].predict(total_df[self.locIndices])
        total_df = self.descale(total_df)
        return total_df

    def rescale(self, data : pd.DataFrame) -> pd.DataFrame:
        """rescales the data, since gprs like them to be normalized

        Args:
            data (pd.DataFrame): data to normalize. Columns must match the df used for fitting!

        Returns:
            pd.DataFrame: return the rescaled data
        """
        return (data - self.min)/(self.max - self.min)[self.locIndices]
    
    def descale(self, data : pd.DataFrame) -> pd.DataFrame:
        """Reverses the effect of the descale method.

        Args:
            data (pd.DataFrame): expects a normalized df, returned after prediction from the GPRs

        Returns:
            pd.DataFrame: _description_
        """
        return (self.max - self.min) * data + self.min
    
    @staticmethod
    def getGPR() -> GaussianProcessRegressor:
        """A default kernel that works well enough here. nu = 1.5 and 2.5 might work well too,
        other values of nu would be expensive to compute

        Returns:
            GaussianProcessRegressor: an sklearn GPR object
        """
        kernel = WhiteKernel() + Matern(length_scale=1.0, nu = 0.5)
        return GaussianProcessRegressor(kernel = kernel)
    
    @staticmethod
    def processFile(filepath : str, time_col_name : str) -> pd.DataFrame:
        """A method to process and clean data for subsequent fitting.
        Specifically, it changes the str datetime column of wind_data.csv
        to represent a column of ints corresponding to the hourly timedelta
        since the first time.
        Args:
            filepath (str): path to wind_data.csv
            time_col_name (str): column name of the one containing the time data

        Returns:
            pd.DataFrame: a dataframe containing the modified data
        """
        raw_df = pd.read_csv(filepath)
        #convert to datetime object to enable mathematical operations
        raw_df[time_col_name] = pd.to_datetime(raw_df[time_col_name])

        #convert the date column to an integer valued column representing hours since start
        first_time = raw_df[time_col_name].unique()[0]
        raw_df["times"] = (raw_df[time_col_name] - first_time).dt.components.hours
        #we need to account for the next day as a special case
        #since its hour difference is 0 from start time
        mask = (raw_df[time_col_name] - first_time).dt.days == 1 
        raw_df.loc[mask, "times"] = 24
        raw_df = raw_df.drop(columns = [time_col_name], axis = 1)

        return raw_df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filepath = "../data/wind_data/wind_data.csv"
    raw_df = SpatialInterpolator.processFile(filepath, "ob_end_time")
    interp = SpatialInterpolator(raw_df, ["station_x", "station_y", "times"], ["horizontal_wind_speed", "vertical_wind_speed"])
    interp.fit()
    #with open( 'model.pkl','wb') as f:
    #    pickle.dump(interp,f)
    with open("model.pkl", "rb") as f:
        interp = pickle.load(f)
    "example usage"
    if True:
        nodes = np.loadtxt('../data/esw_grids/esw_nodes_50k.txt')
        nodes = nodes.T
        df = interp.interpolate(nodes, [43200.1/3600])
        fig, ax = plt.subplots( figsize = (5,10))
        plt.cla()
        plt.quiver(df["station_x"], df["station_y"], df["horizontal_wind_speed"].to_numpy(), df["vertical_wind_speed"].to_numpy())
        plt.show()

