import os
import sys
import zipfile

from io import open

def prepare_text8():
    if os.path.exists('./data/text8/train.txt'):
        print('Tokenized text8 already exists - skipping processing')
        return()

    data = zipfile.ZipFile('./data/text8/text8.zip').extractall('./data/text8')
    data = open('./data/text8/text8', 'r').read()
    # if we read above in utf-8, it creates a space after each char
    print('Length of text8: {}'.format(len(data)))

    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    for fn, part in [('./data/text8/train.txt', train_data), ('./data/text8/valid.txt', valid_data), ('./data/text8/test.txt', test_data)]:
        print('{} will have {} bytes'.format(fn, len(part)))
        print('- Tokenizing...')
        # Change space ' ' to underscore '_'
        #part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])
        part_str = ' '.join([c  for c in part.strip()])
        print('- Writing...')
        f = open(fn, 'w').write(part_str)
        f = open(fn + '.raw', 'w', encoding='utf-8').write(part)
    print('text8 data files are ready for further processing..')
