import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, ElasticNet, ElasticNetCV
from application_logger.logger import Logger

# Executing the logger class
logger_obj = Logger(
    logger_name=__name__, file_name=__file__, streamLogs=True)
lgr = logger_obj.get_logger()


class LModel:
    """
    Class specific to build a regression model to regress against the label column 'Air Temperature [K]'
    """

    def __init__(self):

        self.df = pd.read_csv("ai4i2020.csv")
        self.X = None
        self.Y = None

        self.lambda1 = None
        self.lambda2 = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.scaler = StandardScaler()  # an object of StandardScaler()

        self.lModel = None
        self.l1Model = None
        self.l2Model = None
        self.elasticModel = None

        lgr.info("Model building initiates..")

    def _normalize(self, data):
        """
        This method tries to normalize the argumented data using Box-cox transformation and since box-cox transformation
        internally essentially takes the logarithm of the data for lambda = 0, so this function will take care of any non-positive
        values and replace them by the median of that very data.
        """
        try:
            data_normal, lambdA = stats.boxcox(data)
            return data_normal, lambdA

        except ValueError as ve:
            # replacing all the non-positive values by the median of that very column
            data_ = data.mask(data <= 0, data.median())
            return self._normalize(data_)

        except Exception as e:
            lgr.error("LModel._normalize()", e)

    def _standardize(self, data):
        """
        This method standardizes the data without changing its meaning per se, so as to increase the model optimization.
        """
        try:
            return self.scaler.fit_transform(data)

        except Exception as e:
            lgr.error("LModel._standardize()", e)

    def _features(self):
        """
        A protected method specific to pick the feature columns from the datset.
        """
        try:
            self.X = self.df.drop(
                columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]'])

            """
            since we know from our above analysis, 'Rotational speed [rpm]' and 'Tool wear [min]'
            are not normal so it's better if we normalize them here only.
            """
            self.X['Rotational speed [rpm]'], self.lambda1 = self._normalize(
                self.X['Rotational speed [rpm]'])
            self.X['Tool wear [min]'], self.lambda2 = self._normalize(
                self.X['Tool wear [min]'])

            return self.X

        except Exception as e:
            lgr.error("LModel._features()", e)

    def _label(self):
        """
        A protected method specific to select the label column that is to be regressed against.
        """
        try:
            self.Y = self.df[['Air temperature [K]']]
            return self.Y

        except Exception as e:
            lgr.error("LModel._label()", e)

    def _split(self):
        try:
            # standardizing the features before splitting
            X_ = self._standardize(self._features())
            self._label()

            # splitting the data into train and test sub-data
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                X_, self.Y, test_size=0.25, random_state=100)

        except Exception as e:
            lgr.error("LModel._split()", e)

    def build(self):
        """
        A method specific to build the desired model.
        """
        try:
            self._split()  # so that our test and train datasets are ready to be fed

            self.lModel = LinearRegression()

            lgr.info("readying the model..")
            self.lModel.fit(self.X_train, self.Y_train)
            lgr.info("Model executed succesfully!")

        except Exception as e:
            lgr.error("LModel.build()", e)

    def buildLasso(self):
        """
        This method builds the model with Lasso Regression regularization so that error terms are in more control. 
        """
        try:
            self._split()

            # cross validation to compute the best value of alpha
            lassocv = LassoCV(alphas=None, cv=10, max_iter=1000)
            lassocv.fit(self.X_train, self.Y_train)

            lgr.info("readying the L1 Model...")
            self.l1Model = Lasso(lassocv.alpha_)
            self.l1Model.fit(self.X_train, self.Y_train)
            lgr.info("L1 Model executed!")

        except Exception as e:
            lgr.error("LModel.buildLasso()", e)

    def buildRidge(self):
        """
        This method builds the model with Ridge Regression Regularization so that error terms are in more control. 
        """
        try:
            self._split()

            # cross validation to compute the best value of alpha
            ridgecv = RidgeCV(cv=10)
            ridgecv.fit(self.X_train, self.Y_train)

            lgr.info("readying the L2 Model...")
            self.l2Model = Ridge(ridgecv.alpha_)
            self.l2Model.fit(self.X_train, self.Y_train)
            lgr.info("L2 Model executed!")

        except Exception as e:
            lgr.error("LModel.buildRidge()", e)

    def buildElasticNet(self):
        """
        This method builds the model with ElasticNet Regression regularization so that error terms are in more control. 
        """
        try:
            self._split()

            # cross validation to compute the best value of alpha
            en_cv = ElasticNetCV(cv=10)
            en_cv.fit(self.X_train, self.Y_train)

            lg.info("readying the ElasticNet Model...")
            self.elasticModel = ElasticNet(en_cv.alpha_)
            self.elasticModel.fit(self.X_train, self.Y_train)
            lg.info("ElasticNet Model executed!")

        except Exception as e:
            lgr.error("LModel.builkdElasticNet()", e)

    def accuracy(self, mode='Regression'):
        """
        This method calculates the accuracy of the built model based on the `Adjusted R-squared`.
        """
        try:
            if mode == 'Elastic':
                r_sq = self.elasticModel.score(self.X_test, self.Y_test) * 100

            elif mode == 'L1':
                r_sq = self.l1Model.score(self.X_test, self.Y_test) * 100

            elif mode == 'L2':
                r_sq = self.l2Model.score(self.X_test, self.Y_test) * 100

            else:
                r_sq = self.lModel.score(self.X_test, self.Y_test) * 100

            n = self.X_test.shape[0]               # number of rows
            p = self.X_test.shape[1]               # number of predictors
            accuracy_ = 1-(1 - r_sq)*(n-1)/(n-p-1)  # adjusted r-squared

            lgr.info(f"The {mode} model appears to be {accuracy_}% accurate.")
            return round(accuracy_, 3)

        except Exception as e:
            lgr.error("LModel.accuracy()", e)

    def predict(self, process_t, rot_speed, torque, tool_wear, machine_failure, twf, hdf, pwf, osf, rnf, mode="Regression"):
        """
        A method specific to yield prediction results.

        Note: Even if one of the twf, hdf, pwf, osf and rnf failures is true, then Machine failure will be set to 1.
              For further clarification, refer to the dataset desrciption.

        However, whatever we made happen to the test data, the same's gotta be done for the input to yield accurate
        prediction outcome that is to say first normalization and then the standardization.
        """
        try:
            # correcting machine_failure if the need arises
            machine_failure = 0
            failures = [twf, hdf, pwf, osf, rnf]
            for i in failures:
                if i == 1:  # i.e if any of these failures become true
                    machine_failure = 1  # set the machine failure = true
                    break

            # the record we passed as arguments
            test_input = ([process_t, rot_speed, torque, tool_wear,
                          machine_failure, twf, hdf, pwf, osf, rnf])

            # normalizing the values of the fetaures that were non-normal

            def n_conversion(t_input, feature, lambdA):
                try:
                    if t_input <= 0:
                        # replace by median, if the value is <= 0
                        t_input = self.df[feature].median

                    # applying box-cox formula
                    if (lambdA == 0):
                        return np.log(t_input)
                    else:
                        return np.power(t_input, lambdA)-1/lambdA

                except Exception as e:
                    lgr.error(e)

            # normalization using the function that I just made
            test_input[1] = n_conversion(
                test_input[1], "Rotational speed [rpm]", self.lambda1)
            test_input[3] = n_conversion(
                test_input[3], "Tool wear [min]", self.lambda2)

            # conversion of input passed into array and reshaping it to be fed into StandardScaler()
            test_input = np.array(test_input).reshape(1, -1)

            # standardize
            std_test_input = self.scaler.transform(test_input)

            if mode == "L1":
                return float(self.l1Model.predict(std_test_input))
            elif mode == "L2":
                return float(self.l2Model.predict(std_test_input))
            elif mode == "Elastic":
                return float(self.elasticModel.predict(std_test_input))
            else:
                return float(self.lModel.predict(std_test_input))

        except Exception as e:
            lgr.error("LModel.predict()", e)

    def save(self, mode="Regression"):
        """
        The method to save the model locally.
        """
        try:
            if mode == "ElasticNet":
                pickle.dump(self.elasticModel, open(
                    "predictive_maintenance.sav", 'wb'))
            elif mode == "L1":
                pickle.dump(self.l1Model, open(
                    "predictive_maintenance.sav", 'wb'))
            elif mode == "L2":
                pickle.dump(self.l2Model, open(
                    "predictive_maintenance.sav", 'wb'))
            else:
                pickle.dump(self.lModel, open(
                    "saved_models/model.sav", 'wb'))

            lgr.info(f"The {mode} model is saved at {os.getcwd()} sucessfully!")

        except Exception as e:
            lgr.error("LModel.save()", e)
