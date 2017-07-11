import csv
import config

subset = {}
with open('data/LibriSpeech/SPEAKERS.TXT') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='|')
    for row in spamreader:
    	if row[0].startswith(';'):
    		continue
    	if row[2].strip() == config.data_subset:
    		subset[row[0].strip()] = {'sex':row[1].strip()}

for key in subset:
	subset[key]['chapters'] = [];

with open('data/LibriSpeech/CHAPTERS.TXT') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='|')
    for row in spamreader:
    	if row[0].startswith(';'):
    		continue
    	if row[3].strip() == "dev-clean":
    		subset[row[1].strip()]['chapters'].append(row[0].strip())

for (key, elements) in subset.items():
	print(key,'  ',elements)