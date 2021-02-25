#the code in this file is created following: https://github.com/abisee/pointer-generator/blob/b29e986f24fdd01a6b6d6008187c5c887f0be282/decode.py#L201
import os
import os.path as p
import pyrouge
from Utils import *

model_type = "BART"
base = "/home/ubuntu/CNNDM/" + model_type + "/"
ref_list_location = base + "/test_only_summary_list.pkl"
generated_list_location = base+model_type+"/generated_summaries_list.pkl"

ref_dir = base + 'references/'
generated_dir = base + 'generated/'

if p.exists(ref_dir) == False:
    os.mkdir(ref_dir)
if p.exists(generated_dir) == False:
    os.mkdir(generated_dir)

def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s

def write_for_rouge(reference_sents, generated_summary, ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.
    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_words = generated_summary.split(' ')
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:  # there is text remaining that doesn't end in "."
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
        decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(generated_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx,sent in enumerate(reference_sents):
            f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
        for idx,sent in enumerate(decoded_sents):
            f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    #logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)

def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.
  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  with open(results_file, "w") as f:
    f.write(log_str)

ref_list = load_data(ref_list_location)
gen_list = load_data(generated_list_location)

i = 0
for ref_summary, gen_summary in zip(ref_list,gen_list):
    write_for_rouge(ref_summary, gen_summary, i)
    i+=1

rouge_result_dict = rouge_eval(ref_dir,generated_dir)

rouge_log(rouge_result_dict, base)
