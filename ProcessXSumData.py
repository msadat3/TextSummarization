import pandas
from datasets import load_dataset

def remove_new_lines_for_BART(text):
    #text = text.replace(' .\n', '. ')
    #text = text.replace('\n', ' ')
    #print(text)
    text = str(text)
    text = text.replace('\n', ' ')
    return text

def remove_new_lines_for_BART_all(file_location, output_location):
    df = pandas.read_csv(file_location)
    df['source'] = df['source'].apply(remove_new_lines_for_BART)
    df['summary'] = df['summary'].apply(remove_new_lines_for_BART)
    df.to_csv(output_location)

def create_only_summary_txt(output_location,test_df_loc):
    test_data = pandas.read_csv(test_df_loc)
    with open(output_location, 'w', encoding="utf-8") as output_file:
        for summary in test_data['summary']:
            output_file.write(summary + '\n')

'''base = 'F:\ResearchData\Summarization\XSum_dataset\\'
remove_new_lines_for_BART_all(base+'train.csv', base+'BART\\train.csv')
remove_new_lines_for_BART_all(base+'test.csv', base+'BART\\test.csv')
remove_new_lines_for_BART_all(base+'validation.csv', base+'BART\\validation.csv')

create_only_summary_txt(base+'BART\\test_only_summary.txt', base+'BART\\test.csv')'''

############from huggingface ##########
base = "F:\ResearchData\Summarization\Xsum_huggingface\\"
def convert_to_csv(split, outputLocation):
    ids = dataset[split]['id']
    sources = dataset[split]['document']
    summaries = dataset[split]['summary']

    DataFrame = pandas.DataFrame({'source': sources, 'summary': summaries, 'id': ids})
    DataFrame.to_csv(outputLocation)
    print(split, DataFrame.shape)

dataset = load_dataset("xsum")
#convert_to_csv('test', base+'test.csv')
#convert_to_csv('validation', base+'validation.csv')
convert_to_csv('train', base+'train.csv')

#print('test')
#remove_new_lines_for_BART_all(base+'test.csv', base+'BART\\test.csv')
#print('valid')
#remove_new_lines_for_BART_all(base+'validation.csv', base+'BART\\validation.csv')
print('train')
remove_new_lines_for_BART_all(base+'train.csv', base+'BART\\train.csv')

#create_only_summary_txt(base+'BART\\test_only_summary.txt', base+'BART\\test.csv')