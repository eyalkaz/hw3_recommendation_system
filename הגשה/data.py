# Eyal Kazula 209133693
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    "*** YOUR CODE HERE ***"
    ratings = data[0]
    # 1.1
    print("unique users: ", ratings["userId"].nunique())
    print("unique movies: ", ratings["movieId"].nunique())
    print("number of ratings: ", ratings["userId"].count())
    # 1.2
    rating_per_movie = ratings["movieId"].value_counts()
    print("max movie ratings:", rating_per_movie.max())
    print("min movie ratings:", rating_per_movie.min())
    # 1.3
    ratings_per_user = ratings["userId"].value_counts()
    print("max user ratings:", ratings_per_user.max())
    print("min user ratings:", ratings_per_user.min())


def plot_data(data, plot = True):
    "*** YOUR CODE HERE ***"
    # get mean rating of all movies
    ratings = data[0]
    grades_series = ratings["rating"].value_counts().sort_index()
    grades_series.plot.bar(logy=True, title="log scale rating")
    plt.xlabel("Rating")
    plt.ylabel("Log appearances")
    if plot:
        plt.show()
