import csv
import config

def read_data_header(subset_name=config.data_subset):      
    subset = {}
    with open('data/LibriSpeech/SPEAKERS.TXT') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|')

        for row in spamreader:

            if row[0].startswith(';'): 
                continue
            if row[2].strip() == subset_name:
                subset[row[0].strip()] = dict({'sex':row[1].strip()})
                subset[row[0].strip()]['chapters'] = []
    with open('data/LibriSpeech/CHAPTERS.TXT') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|')
        for row in spamreader:
        	if row[0].startswith(';'):
        		continue
        	if row[3].strip() == subset_name:
        		subset[row[1].strip()]['chapters'].append(row[0].strip())
    print subset
    return subset




