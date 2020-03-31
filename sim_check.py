import multiprocessing
import time
import os
import sys
import argparse
import queue
import gensim
import tempfile
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from UniversalCLI.CLI import CLI
from UniversalCLI.Components import *

source_code_file_extensions = [".py", ".ipynb"]
file_column_label = "File"
similarity_column_label = "Similarity (%)"
similarity_label_length = len(similarity_column_label)

def gen_doc(source_code, file, queue):
    queue.put([file, [word.lower() for word in word_tokenize(source_code[file])]])

def check_sim(source_file, source_code, dictionary, tf_idf, largest_string_length, sims):

    query_doc = [w.lower() for w in word_tokenize(source_code[source_file])]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]

    re = ""
    re += "\n" + "Duplication probability - " + source_file + "\n"
    re += "-" * (largest_string_length + similarity_label_length) + "\n"
    re += "%s %s" % (file_column_label.center(largest_string_length), similarity_column_label) + "\n"
    re += "-" * (largest_string_length + similarity_label_length) + "\n"

    for similarity, source in zip(sims[query_doc_tf_idf], source_code):
        # Ignore similarities for the same file
        if source == source_file:
            continue
        similarity_percentage = similarity * 100
        text = "GOOD!" if similarity_percentage < 20 else (
            "OK!" if similarity_percentage < 40 else (
            "NOT GOOD!" if similarity_percentage < 70 else "SO BAD!"))
        re += "%s     " % (source.ljust(largest_string_length)) + text + "%.2f" % (similarity_percentage) + "\n"
    return re

def main():

    parser_description = "=== Duplicate Code Detection Tool ~ Kira ==="
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument('-o', '--output', help="Name an output file where similarities between files will be stored.")
    parser.add_argument('-w', '--worker', help="Indicate number of workers used.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--directory",
                       help="Check for similarities between all files of the specified directory.")
    group.add_argument('-f', "--files", nargs="+", help="Check for similarities between specified files. \
                        The more files are supplied the more accurate are the results.")
    
    args = parser.parse_args()

    # Determine which files to compare for similarities
    source_code_files = list()
    if args.directory:
        if not os.path.isdir(args.directory):
            print("[Warning] Path does not exist or is not a directory:", args.directory)
            sys.exit(1)
        # Get a list with all the source code files within the directory
        for dirpath, _, filenames in os.walk(args.directory):
            for name in filenames:

                _, file_extension = os.path.splitext(name)
                if file_extension in source_code_file_extensions:
                    filename = os.path.join(dirpath, name)
                    source_code_files.append(filename)
    else:
        if len(args.files) < 2:
            print("[Warning] Too few files to compare, you need to supply at least 2")
            sys.exit(1)
        for supplied_file in args.files:
            if not os.path.isfile(supplied_file):
                print("[Warning] Supplied file does not exist:", supplied_file)
                sys.exit(1)
        source_code_files = args.files

    if args.output is None:
        print("[Warning] Path does not exist or is not a valid file name:", args.output)
        sys.exit(1)
    else:
        output_file = args.output
    
    numWorker = 1
    if args.worker is None or int(args.worker) > os.cpu_count() - 2:
        numWorker = os.cpu_count() - 2
    else:
        numWorker = int(args.worker)

    # Parse the contents of all the source files
    source_code = OrderedDict()
    for source_code_file in source_code_files:
        with open(source_code_file, 'r', encoding='UTF-8') as f:
            # Store source code with the file path as the key
            source_code[source_code_file] = f.read()

    # Initialize UCLI
    b1 = ProgressBar(tot=len(source_code), title = "Build Obj", preset='shades', info = lambda cur, tot: '%6.2f' % (cur / tot * 100.0))
    b2 = ProgressBar(tot=len(source_code), title = "Sim Check", preset='shades', info = lambda cur, tot: '%6.2f' % (cur / tot * 100.0))
    fileNumber = CLIComponent(align = 'left')
    currentTask = CLIComponent(align='left')
    overallStatus = CLIComponent(align = 'center')
    CLI.init([[fileNumber], [currentTask], [b1], [b2], [overallStatus]], title='Similarity check', borderstyle = ('magenta', None, ['bold']))

    fileNumber.setContent('Total of %d files detected' % len(source_code))
    overallStatus.setContent('Initializing...')
    CLI.update()
    CLI.render()

    # Create a Similarity object of all the source code
    currentTask.setContent("Current Task: Create a Similarity object of all the source code")

    CLI.log("[Output] Creating processes pool of " + str(os.cpu_count() - 2))
    pool = multiprocessing.Pool(processes=(os.cpu_count() - 2))
    gen_docs = []
    temp2 = multiprocessing.Manager().Queue()
    temp = []
    for file in source_code:
        pool.apply_async(gen_doc, args=(source_code, file, temp2))

    CLI.log('[Output] waiting process to end...')
    pool.close()
    pool.join()
    CLI.log('[Output] all process stop!')
    while not temp2.empty():
        v = temp2.get()
        temp.append(v)
    
    b1.updateProgress(b2.tot)
    overallStatus.setContent('All Done!')
    
    CLI.log('[Output] temp length = %d' % len(temp))
    gen_docs = []
    for file in source_code.keys():
        for item in temp:
            if item[0] == file:
                gen_docs.append(item[1])
                temp.remove(item)

    # gen_docs = [[word.lower() for word in word_tokenize(source_code[source_file])]
    #             for source_file in source_code]

    #print(source_code.items())
    #CLI.log('' + source_code.keys())
    CLI.log('[Output] gen_docs length = ' + str(len(gen_docs)))
    

    

    currentTask.setContent("Current Task: Create a Dictionary")
    dictionary = gensim.corpora.Dictionary(gen_docs)
    currentTask.setContent("Current Task: Create a Corpus")
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    currentTask.setContent("Current Task: Create a Tf-idf")
    tf_idf = gensim.models.TfidfModel(corpus)
    currentTask.setContent("Current Task: Create a Similarity")
    sims = gensim.similarities.Similarity(tempfile.gettempdir() + os.sep, tf_idf[corpus],
                                          num_features=len(dictionary))
    CLI.log('[Output] Code files: ' + str(source_code_files))
    largest_string_length = len(max(source_code_files, key=len))

    currentTask.setContent("Current Task: Check for similarities")
    CLI.log("[Output] Creating processes pool of " + str(os.cpu_count() - 2))
    pool2 = multiprocessing.Pool(processes=(os.cpu_count() - 2))
    results = []
    multiple_res = [pool2.apply_async(check_sim, args=(file, source_code, dictionary, tf_idf, largest_string_length, sims)) for file in source_code]

    while len(pool2._cache) != 0:
        overallStatus.setContent('%d jobs remaining...' % len(pool2._cache))
        for res in multiple_res:
            if res.ready():
                re = res.get()
                b2.updateProgressInc(1)
                results.append(re)
                CLI.log(re)
                multiple_res.remove(res)
        #time.sleep(0.5)
    
    CLI.log('[Output] wait process to end')
    
    pool2.close()
    pool2.join()
    b2.updateProgress(b2.tot)
    CLI.log('[Output] process stop')
    overallStatus.setContent('All Done!')

    with open(output_file, "a") as myfile:
        for result in results:
            myfile.write(result)
            #CLI.log(result)

if __name__ == "__main__":
    
    main()
    
