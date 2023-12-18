# packages
import xml.sax  # parse xml
import numpy as np
import pandas as pd
import mwparserfromhell  # parse wikimedia
from tqdm.notebook import tqdm  # progress bars
from geopy import distance
from data_processor import *  # parse coordinates

# builtins
import os
import sys
import bz2
import gc
import json
from multiprocessing import Pool
import math


def process_article(title, text, timestamp, coord_range_lat, coord_range_long, template="coord"):
    """Process wikipedia article looking for templates and returning them"""

    # Create a parsing object
    wikicode = mwparserfromhell.parse(text)

    # Search through templates for the template
    coord_matches = wikicode.filter_templates(matches=template)

    # Filter out errant matches
    coord_matches = [x for x in coord_matches if x.name.strip_code().strip().lower() == template.lower()]

    # check if match contains coordinates
    if len(coord_matches) >= 1:

        # extract coordinates
        coords = extract_coordinates(str(coord_matches[0]))

        # coords have wrong format
        if not coords:
            return None

        # check if coordinates are in given region
        if coord_range_lat[0] < coords[0] < coord_range_lat[1] and coord_range_long[0] < coords[1] < coord_range_long[1]:

            # Extract all templates
            all_templates = wikicode.filter_templates()

            infobox = [x for x in all_templates if "infobox" in x.name.strip_code().strip().lower()]

            if len(infobox) >= 1:
                # Extract information from infobox if existing
                properties = {param.name.strip_code().strip(): param.value.strip_code().strip()
                              for param in infobox[0].params
                              if param.value.strip_code().strip()}
            else:
                properties = None

            text = wikicode.strip_code().strip()

            # Extract internal wikilinks
            wikilinks = [x.title.strip_code().strip() for x in wikicode.filter_wikilinks()]

            # Extract external links
            exlinks = [x.url.strip_code().strip() for x in wikicode.filter_external_links()]

            # Find approximate length of article
            text_length = len(wikicode.strip_code().strip())

            return [title, coords, properties, text, wikilinks, exlinks, text_length]
        else:
            # object not in given region, disregard
            return None


class SimpleWikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text', 'timestamp'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._pages.append((self._values['title'], self._values['text']))


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Parse through XML data using SAX"""

    def __init__(self, coord_range_lat, coord_range_long):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._articles = []
        self._article_count = 0
        self._non_matches = []
        self.coord_range_lat = coord_range_lat
        self.coord_range_long = coord_range_long

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text', 'timestamp'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._article_count += 1

            # if self._usa:
            #     # rough coordinates of the USA
            #     COORD_RANGE_LAT = (25.11667, 49.040000)
            #     COORD_RANGE_LONG = (-125.666666, -59.815000)
            # else:
            #     # rough coordinates of Allegheny region
            #     COORD_RANGE_LAT = (40.000000, 40.870000)
            #     COORD_RANGE_LONG = (-80.550000, -79.500000)

            # Search through the page to see if the coordinate is in the Allegheny region
            article = process_article(**self._values, coord_range_lat=self.coord_range_lat,
                                      coord_range_long=self.coord_range_long)
            # Append to the list of articles
            if article:
                self._articles.append(article)


def find_articles(input, limit=None, save=True):
    """Find all the articles in specified region from a compressed wikipedia XML dump.
       `limit` is an optional argument to only return a set number of articles.
        If save, articles are saved to a partition directory based on file name"""

    data_path, region, coord_range_lat, coord_range_long = input

    # Object for handling xml
    handler = WikiXmlHandler(coord_range_lat, coord_range_long)

    # Parsing object
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    # Iterate through compressed file
    for i, line in enumerate(bz2.BZ2File(data_path, 'r')):
        try:
            parser.feed(line)
        except StopIteration:
            break

        # Optional limit
        if limit is not None and len(handler._articles) >= limit:
            return handler._articles

    if save:
        partition_dir = os.path.dirname(os.path.dirname(data_path)) + f"\\uncompressed_{region}"

        if not os.path.exists(partition_dir):
            os.mkdir(partition_dir)

        # Create file name based on partition name
        p_str = data_path.split('-')[-1].split('.')[-2]
        out_dir = partition_dir + f'\\{p_str}.ndjson'

        # Open the file
        with open(out_dir, 'w') as fout:
            # Write as json
            for article in handler._articles:
                fout.write(json.dumps(article) + '\n')
        print(f'{len(os.listdir(partition_dir))} files processed.', end='\r')

    # Memory management
    del handler
    del parser
    gc.collect()
    return None


def process(compressed_path, region, coord_range_lat, coord_range_long, cores):
    """Decompress all articles in compressed_path, look for templates and save articles fitting criteria"""

    partitions = [(compressed_path + file, region, coord_range_lat, coord_range_long) for file in os.listdir(compressed_path) if 'xml-p' in file]
    len(partitions), partitions[-1]

    uncompressed_path = os.path.dirname(os.path.dirname(compressed_path)) + f"\\uncompressed_{region}"
    if not os.path.exists(uncompressed_path):
        os.mkdir(uncompressed_path)

    if not len(os.listdir(uncompressed_path)) > 50:  # check if already preprocessed
        pool = Pool(processes=cores)
        results = []

        # Run partitions in parallel
        for x in tqdm(pool.imap(find_articles, partitions), total=len(partitions)):
            results.append(x)

        pool.close()
        pool.join()
    else:
        print("Articles already processed.")


def create_dist_cols(row, category, df_category, max_dist):
    """Returns row with added distance based features"""

    # create new dataframe with dist to every gis location
    df_category_copy = df_category.loc[:, ["coords"]].copy()
    df_category_copy["prop_coords"] = [[row["lat"], row["long"]]]*df_category_copy.shape[0]
    # calculate distance from current property to all objects
    df_category_copy["dist"] = df_category_copy.apply(lambda x: distance.distance(x["coords"],
                                                                                  x["prop_coords"]).m, axis=1)

    count = df_category_copy[df_category_copy["dist"] < max_dist].shape[0]  # count closer than max_dist m away
    closest = df_category_copy["dist"].min()  # minimum distance to object for this category

    row[category + "_count"] = count
    row[category + "_dist"] = closest

    return row


def add_dist_features(input):
    """Multiprocessing service for process_dist_features"""

    df_structured, categories_dfs, max_dist = input

    for category in categories_dfs:
        df_structured[category + "_dist"] = math.inf
        df_structured[category + "_count"] = 0
        df_category = categories_dfs[category]

        df_structured = df_structured.apply(create_dist_cols, axis=1, result_type="broadcast",
                                            args=[category, df_category, max_dist])

    gc.collect()  # memory management
    return df_structured


def process_dist_features(df_structured, categories_dfs, max_dist, cores):
    """Returns structured df with added distance based features"""

    partitions = np.array_split(df_structured, 100)  # divide dataframe into 100 parts
    partitions = [(df, categories_dfs, max_dist) for df in partitions]  # add reference to category data

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap(add_dist_features, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


def create_text_cols(row_outer, places_df, embedding, feature_names, max_dist, weighted, mean=True, k_closest=math.inf, weight_func="tanh"):
    """Returns row along with mean of weighted word embeddings for articles closer than max_dist for property in row"""

    in_range = []
    dists = []
    for index, row in places_df.iterrows():
        dist = distance.distance(row["coords"], [row_outer["latitude"], row_outer["longitude"]]).m
        if dist < max_dist:
            # article is closer than max_dist m
            if weight_func != "None":
                base_weight = 1 - (dist / max_dist)  # calculate weight between 0 and 1
                if weight_func == "linear":
                    weight_trans = base_weight  # linear
                elif weight_func == "cubic":
                    weight_trans = base_weight**3  # cubic
                else:
                    weight_trans = -np.exp(-((base_weight+0.3)**6))+1  # tanh
                in_range.append(embedding[index] * weight_trans)  # multiply embedding by weight
            else:
                in_range.append(embedding[index])  # <- NO WEIGHTING
            dists.append(dist)

    if len(in_range) == 0:
        # no articles found inside max_dist radius
        row_outer["article_count"] = len(in_range)
        in_range = [pd.NA] * len(embedding[0])
        row_outer.loc[feature_names[0]:feature_names[-1]] = in_range

    else:
        if len(in_range) > k_closest:
            # more than 100 articles found inside max_dist radius
            # sort by distance descending (lowest distance last)
            in_range_sorted = [vec for vec, _ in sorted(zip(in_range, dists), key=lambda x: x[1], reverse=True)]
            in_range = in_range_sorted[:k_closest]  # cutoff at k_closest
            # print("cutoff applied")

        # calculate mean of weighted embeddings and assign to word columns
        article_count = len(in_range)
        if mean:
            # print(f"in_range: {len(in_range)}")
            # print(f"row_loc: {row_outer.loc[feature_names[0]:feature_names[-1]].shape}")
            # print(f"sum_loc: {len([sum(x) / article_count for x in zip(*in_range)])}")
            row_outer.loc[feature_names[0]:feature_names[-1]] = [sum(x) / article_count for x in zip(*in_range)]
        else:
            row_outer.loc[feature_names[0]:feature_names[-1]] = [sum(x) for x in zip(*in_range)]  # just sum
        row_outer["article_count"] = article_count

    return row_outer


def add_text_features(input):
    """Multiprocessing service for process_text_features"""

    df_structured, places_df, embedding, feature_names, max_dist, weighting, mean, weight_func, k_closest = input

    row_count = df_structured.shape[0]
    for feature in feature_names:
        # add column for each word
        df_structured[feature] = [0]*row_count

    # add column for amount of articles found
    df_structured["article_count"] = [0]*row_count

    # defragment df
    df_structured = df_structured.copy()

    df_structured = df_structured.apply(create_text_cols, axis=1, result_type="broadcast",
                                        args=[places_df, embedding, feature_names, max_dist, weighting, mean, k_closest, weight_func])

    gc.collect()  # memory management
    return df_structured


def process_text_features(df_structured, places_df, embedding, feature_names, max_dist, cores, weighting, mean, weight_func, k_closest=100):
    """Returns structured df with added text based features"""

    partitions = np.array_split(df_structured, 100)  # divide dataframe into 100 parts
    # add reference to places and tf data
    partitions = [(df, places_df, embedding, feature_names, max_dist, weighting, mean, weight_func, k_closest) for df in partitions]

    # results = add_text_features(partitions[0])

    pool = Pool(processes=cores)
    results = []

    # Run partitions in parallel
    for x in tqdm(pool.imap(add_text_features, partitions), total=len(partitions)):
        results.append(x)

    pool.close()
    pool.join()

    return pd.concat(results, ignore_index=True)


# wrap execution in main method to guard multiprocessing code
if __name__ == "__main__":
    __spec__ = None  # remove spec to be able to run this script from iPython (Jupyter Notebook)
    compressed_path = sys.argv[1]
    cores = int(sys.argv[2])
    process(compressed_path, cores)

