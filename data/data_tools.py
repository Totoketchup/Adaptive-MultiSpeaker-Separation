import csv
import config
import os as os

# Read the metadata of LibriSpeech dataset
# Put them in a dictionnary with the following structure
# {
#  [speaker_key:{'sex':M/F, 'chapters':[(int), ...]}, ..., speaker_key:{..},]  
# }
def read_metadata(subset_name=config.data_subset):      
    metadata = {}
    path_speakers = os.path.join(config.workdir, 'data/LibriSpeech/SPEAKERS.TXT')
    path_chapters = os.path.join(config.workdir, 'data/LibriSpeech/CHAPTERS.TXT')

    # Read speakers information
    with open(path_speakers) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|')
        for row in spamreader:
            # It's a comment
            if row[0].startswith(';'): 
                continue
            # Read only keys from the dataset subset
            if row[2].strip() == subset_name:
                # Add the sex information for this key
                metadata[row[0].strip()] = dict({'sex':row[1].strip()})
                # Create the chapters array for next file
                metadata[row[0].strip()]['chapters'] = []

    # read chapters information
    with open(path_chapters) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|')
        for row in spamreader:
            # It's a comment
        	if row[0].startswith(';'):
        		continue
            # Read only keys from the dataset subset
        	if row[3].strip() == subset_name:
                # Add all the chapters for this speaker
        		metadata[row[1].strip()]['chapters'].append(row[0].strip())

    return metadata


# Extract keys with the designated sex
def get_sex_subset(metadata, sex):
    return [k for k, v in metadata.items() if v['sex'] == sex]

# Extract males keys
def males_keys(metadata):
    return get_sex_subset(metadata, 'M')

# Extract females keys
def females_keys(metadata):
    return get_sex_subset(metadata, 'F')