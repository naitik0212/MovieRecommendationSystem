import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from baseline import loadModel
from baseline import baselineEstimate

import os
RANGE_MIN = 1
RANGE_MAX = 2
NUM_SIMILAR = 20
DATASET_ROOT_PATH = os.path.join(os.getcwd(),'./','')
OUT_PUT =[]


def readcsv(name):
    return pd.read_csv(os.path.join(DATASET_ROOT_PATH, name))

def calculateRatingIX(mid,uid):
    similarityMatrix = np.load("similarityMatrix.npy")
    indexMatrix = np.load("movieindex.npy")
    index = np.where(indexMatrix == mid)[0][0]
    similarity = similarityMatrix[index]
    similarityDF =  pd.DataFrame(similarity.reshape(-1,len(similarity)))
    similarityDF = similarityDF.transpose()
    indexDF = pd.DataFrame(indexMatrix.reshape(-1,len(indexMatrix)))
    indexDF = indexDF.transpose()
    newDF = pd.concat([indexDF,similarityDF],axis=1)
    newDF.columns = ['movieId','similarity']

    ratings = readcsv('Data/ml-20m/train_ratings.csv')
    ratingU = ratings.loc[ratings['userId'] == uid]
    ratingU = ratingU.drop('timestamp',1)

    newRating = ratingU.merge(newDF, left_on='movieId', right_on='movieId')
    newRating = newRating.sort_values('similarity', ascending=False)
    top = newRating.head(n=10)

    similaritySum = top['similarity'].sum()
    baselinemodel = loadModel()
    bxi = baselineEstimate(baselinemodel, uid,mid)
    sum = 0
    for index, row in top.iterrows():
        bxj = baselineEstimate(baselinemodel, row['userId'],row['movieId'])
        rxj = row['rating']
        sij = row['similarity']
        sum += (rxj-bxj) * sij

    rxi = bxi + sum/similaritySum

    print(rxi)



# Uses mltags.csv and combines it with move actor on movie id
# genarates term vector, calculates the term frequency along with weights and Inverse Document Frequency,
# TF is stored in a CSV files for all actors
# The IDF is then multiplied with TF and stored in an other csv file

def generate_model():
    print("Generating tags model")

    if os.path.exists("similarityMatrix.npy"):
        similarityMatrix = np.load("similarityMatrix.npy")
    else:
        merged = readcsv('new_tags_generes.csv')
        merged['COUNTER'] = 1
        merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])

        group_data = pd.DataFrame(merged.groupby(['movieId', 'genres'])['COUNTER'].sum())
        term_vector = group_data.pivot_table('COUNTER', ['movieId'], 'genres')
        term_vector.index.names = ['label']
        count_df = generate_idf(term_vector)
        term_vector = term_vector.fillna(0)
        term_vector = generate_TF(term_vector)

        tf_idf = term_vector.copy(deep=True)
        tf_idf = tf_idf.mul(count_df.ix[2], axis='columns')
        indexValues = tf_idf.index.values.tolist()
        print(indexValues)
        print(type(indexValues))
        indexValues = np.array(indexValues)
        np.save("movieindex.npy", indexValues)

        temp = tf_idf.as_matrix()
        svd = TruncatedSVD(n_components=30)
        x = svd.fit_transform(temp)
        similarityMatrix = cosine_similarity(x)
        np.save("similarityMatrix.npy", similarityMatrix)


# combines the given csv files and calculates weight of each tag and generates a term vector

def get_term_vector(df_genome_tags, df_movies_genre, df_tags_with_min_max):
    merged = pd.merge(df_movies_genre, df_tags_with_min_max.iloc[:, [1, 2, 6]],
                      on='movieid')
    merged['COUNTER'] = 1
    merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])
    merged['ts_rank'] = pd.to_numeric(merged['ts_rank'])
    merged['COUNTER'] = merged.COUNTER * merged.ts_rank
    merged = pd.merge(merged, df_genome_tags, left_on='tagid', right_on='tagId')
    group_data = pd.DataFrame(merged.groupby(['movieid', 'tag'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['movieid'], 'tag')
    return term_vector;

# Takes genres, combines the tags with movies ,
# splits the movies which belong to multiple genres into multiple rows,
# genarates term vector for each of the movies, combines it with sum function.
# calculates weight for each tag for the specific genre using the formula given
# multiplies it with the term count of the tag and prints it.

def p1_diff(genre1, genre2):
    print("Generating P1 diff model")
    df_tags = readcsv('mltags.csv')
    df_genome_tags = readcsv('genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')
    df_movies = readcsv('mlmovies.csv')
    df_movies = pd.concat([pd.Series(row['movieid'], row['genres'].split('|'))
                           for _, row in df_movies.iterrows()]).reset_index()
    df_movies.columns = ['genres', 'movieid']
    df_movies_genre1 = df_movies[df_movies['genres'] == genre1]
    df_movies_genre2 = df_movies[df_movies['genres'] == genre2]
    df_movies_genre1_or_genre2 = df_movies[(df_movies['genres'] == genre1) | (df_movies['genres'] == genre2) ]
    df_movies_genre1_or_genre2 = df_movies_genre1_or_genre2.drop_duplicates(subset='movieid',keep = 'first')
    R=len(df_movies_genre1.index)
    M=len(df_movies_genre1_or_genre2.index)
    if R == M:
        print("Can not compare the given genres, as they have same movies")
        return
    genre1_tv = get_term_vector(df_genome_tags, df_movies_genre1, df_tags_with_min_max)
    genre2_tv = get_term_vector(df_genome_tags, df_movies_genre2, df_tags_with_min_max)
    if genre1_tv.empty or genre2_tv.empty:
        print("Can not compare, as one of the genres doesn't exist")
        return
    both_genres_with_duplicates = pd.concat([genre1_tv, genre2_tv])
    genre1_genre_2 = both_genres_with_duplicates.drop_duplicates(keep='first')
    count_df_genre1_and_genre2 = get_column_count(genre1_genre_2)
    count_df_genre1 = get_column_count(genre1_tv)
    count_df_g1_g2_with_duplicates = get_column_count(both_genres_with_duplicates)
    sum_g1 = genre1_tv.sum(axis='index')
    sum_g1_df = sum_g1.to_frame()
    sum_g1_df.columns =['term_count']
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre1, left_index=True, right_index=True, how='outer')
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre1_and_genre2, left_index=True, right_index=True, how='outer')
    sum_g1_df = pd.merge(sum_g1_df, count_df_g1_g2_with_duplicates, left_index=True, right_index=True, how='outer')
    sum_g1_df.columns = ['term_count','r_1j','m_1j','m_1j_with_dups']
    sum_g1_df['r_1j'] = pd.to_numeric(sum_g1_df['r_1j'])
    sum_g1_df['m_1j'] = pd.to_numeric(sum_g1_df['m_1j'])
    sum_g1_df['m_1j_with_dups'] = pd.to_numeric(sum_g1_df['m_1j_with_dups'])
    sum_g1_df = clean_data_frame(sum_g1_df)
    sum_g1_df['weight'] = sum_g1_df.apply(lambda row: get_weight(row,M ,R),
                                          axis=1)

    # with_out_inf = sum_g1_df[sum_g1_df['weight'] != 'inf']
    # max = with_out_inf['weight'].max()
    # sum_g1_df.ix[sum_g1_df['weight'] == 'inf', ['weight']] = [max + 1]
    sum_g1_df['weight'] = pd.to_numeric(sum_g1_df['weight'])
    sum_g1_df['diff'] = sum_g1_df.term_count * sum_g1_df.weight
    #sum_g1_df = sum_g1_df.fillna(0)
    #sum_g1_df = sum_g1_df[sum_g1_df['diff'] != 0]
    sum_g1_df = sum_g1_df[['diff']]
    #print(sum_g1_df.sort_values(by='diff', ascending=0))
    sum_g1_df = sum_g1_df.sort_values(by='diff', ascending=0)
    print(sum_g1_df.to_string())

def get_weight(row, M ,R, is_p1_diff = True):
    # only genre 1 has the tag, hence should be given highest priority
    if is_p1_diff and (row['m_1j'] - row['r_1j']) == 0:
        # one more differenciating factor when m1 is there in both g1 and g2 and only g1
        if row['m_1j'] == row['m_1j_with_dups']:
            return 'inf'
        else:
            return '-inf'
    if is_p1_diff and (M + row['r_1j'] - row['m_1j'] - R) == 0:
        return '-inf'
    # happens in p1 diff when movie m exist in both genres and only this movies has tag t
    if not is_p1_diff and (M + row['r_1j'] - row['m_1j'] - R) == 0:
        return '-inf'
    #if not is_p1_diff and row['m_1j'] - row['r_1j'] == 0:
     #   return 4000 #will never be zero

    if not is_p1_diff and (R - row['r_1j']) == 0:
        #check if M == m if not highly differentiating else no differentiation
        return 'inf'

    return np.log(10+((row['r_1j'] / (R - row['r_1j'])) /
             ((row['m_1j'] - row['r_1j']) / (M + row['r_1j'] - row['m_1j'] - R))))*(np.abs(row['r_1j'] / R - ((row['m_1j'] - row['r_1j']) / (M - R))))


def clean_data_frame(data_frame):
    data_frame = data_frame.fillna(0)
    data_frame = data_frame[data_frame.term_count != 0]
    return data_frame


def get_column_count(term_vector):
    count = term_vector.count(axis='index')
    count_df = count.to_frame()
    return count_df

# Takes genres, combines the tags with movies ,
# splits the movies which belong to multiple genres into multiple rows,
# genarates term vector for each of the movies, combines it with sum function.
# calculates weight for each tag for the specific genre using the formula given
# multiplies it with the term count of the tag and prints it.
def p2_diff(genre1, genre2):
    print("Generating P2 diff model")
    df_tags = readcsv('mltags.csv')
    df_genome_tags = readcsv('genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')
    df_movies = readcsv('mlmovies.csv')
    df_movies = pd.concat([pd.Series(row['movieid'], row['genres'].split('|'))
                           for _, row in df_movies.iterrows()]).reset_index()
    df_movies.columns = ['genres', 'movieid']
    df_movies_genre1 = df_movies[df_movies['genres'] == genre1]
    df_movies_genre2 = df_movies[df_movies['genres'] == genre2]
    df_movies_genre1_or_genre2 = df_movies[(df_movies['genres'] == genre1) | (df_movies['genres'] == genre2)]
    df_movies_genre1_or_genre2 = df_movies_genre1_or_genre2.drop_duplicates(subset='movieid', keep='first')
    R = len(df_movies_genre2.index)
    M = len(df_movies_genre1_or_genre2.index)
    if R == M:
        print("Can not compare the given genres, as they have same movies")
        return
    genre1_tv = get_term_vector(df_genome_tags, df_movies_genre1, df_tags_with_min_max)
    genre2_tv = get_term_vector(df_genome_tags, df_movies_genre2, df_tags_with_min_max)
    if genre1_tv.empty or genre2_tv.empty:
        print("Can not compare, as one of the genres doesn't exist")
        return
    genre1_genre_2 = pd.concat([genre1_tv, genre2_tv])
    genre1_genre_2 = genre1_genre_2.drop_duplicates(keep='first')
    count_df_genre1_and_genre2 = get_column_count(genre1_genre_2)
    count_df_genre1_and_genre2.columns = ['m_1j']
    count_df_genre1_and_genre2['m_1j'] = len(df_movies_genre1_or_genre2) - count_df_genre1_and_genre2.m_1j
    count_df_genre2 = get_column_count(genre2_tv)
    count_df_genre2.columns = ['r_1j']
    count_df_genre2['r_1j'] = len(df_movies_genre2) - count_df_genre2.r_1j
    sum_g1 = genre1_tv.sum(axis='index')
    sum_g1_df = sum_g1.to_frame()
    sum_g1_df.columns = ['term_count']
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre2, left_index=True, right_index=True, how='outer')
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre1_and_genre2, left_index=True, right_index=True, how='outer')
    sum_g1_df['r_1j'] = pd.to_numeric(sum_g1_df['r_1j'])
    sum_g1_df['m_1j'] = pd.to_numeric(sum_g1_df['m_1j'])
    sum_g1_df[['r_1j']] = sum_g1_df[['r_1j']].fillna(value=R)
    sum_g1_df = clean_data_frame(sum_g1_df)
    sum_g1_df['weight'] = sum_g1_df.apply(lambda row: get_weight(row, M, R, False),
                                          axis=1)
    # with_out_inf = sum_g1_df[sum_g1_df['weight'] != 'inf']
    # max = with_out_inf['weight'].max()
    # sum_g1_df.ix[sum_g1_df['weight'] == 'inf', ['weight']] = [max+1]
    sum_g1_df['weight'] = pd.to_numeric(sum_g1_df['weight'])
    sum_g1_df['diff'] = sum_g1_df.term_count * sum_g1_df.weight
    #sum_g1_df =  sum_g1_df[sum_g1_df['diff'] != 0]
    sum_g1_df = sum_g1_df[['diff']]
    #print(sum_g1_df.sort_values(by='diff', ascending=0))
    sum_g1_df = sum_g1_df.sort_values(by='diff', ascending=0)
    print(sum_g1_df.to_string())


def get_model_for_row_id(id, term_vector):
    term_vector.index.name = 'label'
    term_vector.reset_index(inplace=True)
    tv = term_vector[term_vector['label'] == id]
    data = tv.transpose().iloc[1:]
    data.columns = ['term_vector']
    data = data[(data.T != 0).any()]
    return data

def get_model_for_id(id, term_vector):
    try:
        tv = term_vector[term_vector['label'] == id]
        data = tv.transpose().iloc[1:]
        data.columns = ['term_vector']
        data = data[(data.T != 0).any()]
        return data
    except ValueError:
        return None

## given a term vector it generates Term frequency for all the documents
## in the dataframe

def generate_TF(term_vector):
    term_vector['total_freq'] = term_vector.sum(axis=1)
    columns = term_vector.columns.tolist()
    columns.remove('total_freq')
    term_vector = term_vector[columns].div(term_vector.total_freq, axis=0)
    return term_vector

## given a term vector it generates Inverse Document frequency for all the documents
## in the dataframe
def generate_idf(term_vector):
    count = term_vector.count(axis='index')
    count_df = count.to_frame()
    count_df['total_docs'] = len(term_vector.index)
    count_df.columns = ['tag_count', 'total_docs']
    count_df['total_docs'] = pd.to_numeric(count_df['total_docs'])
    count_df['tag_count'] = pd.to_numeric(count_df['tag_count'])
    count_df['idf'] = np.log(count_df.total_docs / count_df.tag_count)
    count_df = count_df.T
    return count_df

# calculates the actor rank.

def scale_down_to_between_two_one(df, field, group_field, new_min_label
                                  , new_max_label, new_field_label):
    min_df = df[field].groupby(df[group_field]).min()
    min_df = min_df.to_frame()
    min_df = min_df.reset_index(level=[group_field])
    max_df = df[field].groupby(df[group_field]).max()
    max_df = max_df.to_frame()
    max_df = max_df.reset_index(level=[group_field])
    min_df.columns = [group_field, new_min_label]
    max_df.columns = [group_field, new_max_label]
    min_max_df = pd.merge(min_df, max_df, on=group_field)
    df_with_min_max = pd.merge(df, min_max_df, on=group_field)
    # df_with_min_max[new_field_label] = df_with_min_max.apply(
    #     lambda row: ((row[field] - row[new_min_label])
    #                  / (row[new_max_label] - row[new_min_label])
    #                  if row[new_max_label] != row[new_min_label] else 1)
    #                 * (RANGE_MAX - RANGE_MIN) + RANGE_MIN, axis=1)
    #
    df_with_min_max[new_field_label] = df_with_min_max.apply(lambda row: (1/row[field]), axis=1)
    return df_with_min_max

# calculates the movie rank
def scale_down_ts_to_between_two_one(df, field, group_field, new_min_label
                                  , new_max_label, new_field_label):
    min_df = df[field].groupby(df[group_field]).min()
    min_df = min_df.to_frame()
    min_df = min_df.reset_index(level=[group_field])
    max_df = df[field].groupby(df[group_field]).max()
    max_df = max_df.to_frame()
    max_df = max_df.reset_index(level=[group_field])
    min_df.columns = [group_field, new_min_label]
    max_df.columns = [group_field, new_max_label]
    min_max_df = pd.merge(min_df, max_df, on=group_field)
    df_with_min_max = pd.merge(df, min_max_df, on=group_field)
    # df_with_min_max[new_field_label] = df_with_min_max.apply(
    #     lambda row: ((row[field].timestamp() - row[new_min_label].timestamp())
    #                  / (row[new_max_label].timestamp() - row[new_min_label].timestamp())
    #                  if row[new_max_label].timestamp() != row[new_min_label].timestamp() else 1)
    #                 * (RANGE_MAX - RANGE_MIN) + RANGE_MIN, axis=1)

    df_with_min_max[new_field_label] = df_with_min_max.apply(lambda row: (1-(1/(1+row[field].timestamp()))), axis=1)
    return df_with_min_max

if not os.path.exists("similarityMatrix.npy"):
    generate_model()

calculateRatingIX(1,1)