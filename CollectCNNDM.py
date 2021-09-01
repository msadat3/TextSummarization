import tensorflow_datasets as tfds
import json
import os
import pandas
from Utils import *

ds = tfds.load('cnn_dailymail',data_dir = 'F:\\ResearchData\\Summarization\\TF_CNNDM\\')

output_dir = 'F:\\ResearchData\\Summarization\\TF_CNNDM\\CSV_format_version_2\\'


if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

def convert_to_json(ds_subset,output_location):
    with open(output_location, 'w', encoding="utf8", ) as location:
        for x in ds_subset:
            json_dict = {}
            json_dict['source'] = x['article'].numpy().decode("utf-8").replace('\\','').strip()
            json_dict['summary'] = x['highlights'].numpy().decode("utf-8").replace('\\','').strip().replace('\n', '<n>')
            json_dict['summary'] = json_dict['summary'].replace(' .','. ')
            json_dict['summary'] = json_dict['summary'].strip()
            location.write(json.dumps(json_dict, ensure_ascii=False) + '\n')

def convert_to_csv(ds_subset,output_location,id_prefix):
    sources = []
    summaries = []
    ids = []
    id = 0
    for x in ds_subset:
        ids.append(id_prefix + '_' + str(id))
        source = x['article'].numpy().decode("utf-8").replace('\\"', '').strip().replace('\n', '')
        if source[:5] == '(CNN)':
            source = source[5:]
        summary = x['highlights'].numpy().decode("utf-8").replace('\\"', '').strip().replace('\n', '<n>')
        summary = summary.replace(' .', '.')
        summary = summary.strip()
        sources.append(source)
        summaries.append(summary)
        id += 1
    DataFrame = pandas.DataFrame({'source': sources, 'summary': summaries, 'id': ids})

    DataFrame = DataFrame.drop_duplicates(subset=['source', 'summary'])
    print(DataFrame.shape)
    DataFrame.to_csv(output_location)
    return DataFrame


def create_only_summary_list(ds_subset,output_location,test_df_loc,id_prefix):
    summaries = []
    #ids = []
    id = 0
    test_data = pandas.read_csv(test_df_loc)
    test_unique_ids = test_data['id'].to_list()
    for x in ds_subset:
        complete_id = id_prefix + '_' + str(id)
        if complete_id in test_unique_ids:
     #   ids.append(id_prefix + '_' + str(id))
            summary = x['highlights'].numpy().decode("utf-8").replace('\\"', '')
            summary_sentences = summary.split('\n')
            summaries.append(summary_sentences)
        else:
            print(id, summary_sentences)
        id += 1
    save_data(summaries,output_location)

def create_only_summary_txt(output_location,test_df_loc):
    test_data = pandas.read_csv(test_df_loc)
    with open(output_location, 'w', encoding="utf-8") as output_file:
        for summary in test_data['summary']:
            output_file.write(summary + '\n')


convert_to_csv(ds['test'],output_dir+'\\test_pegasus.csv', 'test')
convert_to_csv(ds['validation'],output_dir+'\\validation_pegasus.csv', 'validation')
tr = convert_to_csv(ds['train'],output_dir+'\\train_pegasus.csv', 'train')
create_only_summary_txt(output_dir+'\\test_only_summary.txt',output_dir+'\\test.csv')

convert_to_json(ds['test'],output_dir+'\\test.json')
convert_to_json(ds['validation'],output_dir+'\\validation.json')
convert_to_json(ds['train'],output_dir+'\\train.json')


create_only_summary_list(ds['test'],output_dir+'\\test_only_summary_list.pkl',output_dir+'\\test.csv', 'test')