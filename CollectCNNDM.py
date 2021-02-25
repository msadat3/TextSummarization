import tensorflow_datasets as tfds
import json
import os
import pandas
from Utils import *

ds = tfds.load('cnn_dailymail',data_dir = 'F:\\ResearchData\\Summarization\\TF_CNNDM\\')

output_dir = 'F:\\ResearchData\\Summarization\\TF_CNNDM\\CSV_format\\'


if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

def convert_to_json(ds_subset,output_location):
    with open(output_location, 'w', encoding="utf8", ) as location:
        for x in ds_subset:
            json_dict = {}
            json_dict['source'] = x['article'].numpy().decode("utf-8").replace('\\','').strip()
            json_dict['summary'] = x['highlights'].numpy().decode("utf-8").replace('\\','').strip().replace('\n', '')
            location.write(json.dumps(json_dict, ensure_ascii=False) + '\n')

def convert_to_csv(ds_subset,output_location,id_prefix):
    sources = []
    summaries = []
    ids = []
    id = 0
    for x in ds_subset:
        ids.append(id_prefix + '_' + str(id))
        source = x['article'].numpy().decode("utf-8").replace('\\"', '').strip().replace('\n', '')
        summary = x['highlights'].numpy().decode("utf-8").replace('\\"', '').strip().replace('\n', '')
        sources.append(source)
        summaries.append(summary)
        id += 1
    DataFrame = pandas.DataFrame({'source': sources, 'summary': summaries, 'id': ids})

    DataFrame = DataFrame.drop_duplicates(subset=['source', 'summary'])
    print(DataFrame.shape)
    DataFrame.to_csv(output_location)
    return DataFrame


def create_only_summary_list(ds_subset,output_location,id_prefix):
    summaries = []
    #ids = []
    id = 0
    for x in ds_subset:
        id+=1
     #   ids.append(id_prefix + '_' + str(id))
        summary = x['highlights'].numpy().decode("utf-8").replace('\\"', '')
        summary_sentences = summary.split('\n')
        print(id, summary_sentences)
        summaries.append(summary_sentences)
    save_data(summaries,output_location)


create_only_summary_list(ds['test'],output_dir+'\\test_only_summary_list.pkl', 'test')

convert_to_csv(ds['test'],output_dir+'\\test.csv', 'test')
convert_to_csv(ds['validation'],output_dir+'\\validation.csv', 'validation')
tr = convert_to_csv(ds['train'],output_dir+'\\train.csv', 'train')

convert_to_json(ds['test'],output_dir+'\\test.json')
convert_to_json(ds['validation'],output_dir+'\\validation.json')
convert_to_json(ds['train'],output_dir+'\\train.json')
