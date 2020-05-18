import argparse
import re
import os


results_re = re.compile('^checkpoint_(.*)_([0-9]+)\.?[0-9]*.post.results$')
SCORES = ['BLEU', 'chrF++', 'METEOR', 'SemSim']


def argument_parsing():

    # Argument hanlding
    parser = argparse.ArgumentParser(
        description='Organize model results'
    )
    # jbinfo args
    parser.add_argument(
        '--checkpoints',
        type=str,
        default='DATA/models/',
        help='Folder containing checkpoints, config.sh etc'
    )
    parser.add_argument(
        '--link-best',
        action='store_true',
        help='do not link or relink best smatch model'
    )
    parser.add_argument(
        '--no-print',
        action='store_true',
        help='do not print'
    )
    return parser.parse_args()


def read_results(result_files):
    scores = {score: None for score in SCORES}
    with open(result_files) as fid:
        for line in fid:
            for score in SCORES:
                if line.startswith(score):
                    scores[score] = float(line.split()[1])
    return scores 


def collect_results(args, results_regex, main_score):

    # regex of decoding output
    result_path_re = re.compile(
        f'{args.checkpoints}/?([^/]+)/sampling/([^/]+)'
    )

    # Find folders containing results for all epochs for each given model
    epoch_folders = []
    model_folders = []
    for path_pieces in os.walk(args.checkpoints):
        root = path_pieces[0]
        if result_path_re.match(root):
            model_tag, test_tag = result_path_re.match(root).groups()
            epoch_folders.append(
                f'{args.checkpoints}/{model_tag}/sampling/{test_tag}'
            )
            model_folders.append(f'{args.checkpoints}/{model_tag}')

    # loop over each model, loop over results for each epoch, extract results
    # and keep the best BLEU from all epochs
    items = []
    empty_folders = [] 
    for index, epoch_folder in enumerate(epoch_folders):

        # loop each file, I it is an epoch result read it and keep best results
        best_results = {score: None for score in SCORES}
        best_epoch = None
        epochs = []
        for basename in os.listdir(epoch_folder): 
            if results_re.match(basename):
                name, epoch = results_regex.match(basename).groups()
                epochs.append(int(epoch))
                #if 'beam' in epoch_folder:
                #    import ipdb; ipdb.set_trace(context=30)
                results = read_results(os.path.join(epoch_folder, basename))
                if (
                    best_results[main_score] is None or
                    results[main_score] > best_results[main_score]
                ):
                    best_results = results
                    best_epoch = int(epoch)

        if best_results[main_score] is not None:
            # Store data
            items.append({
                'model_folder': model_folders[index],
                'results_folder': epoch_folder,
                f'best_{main_score}_epoch': best_epoch,
                'epochs': sorted(epochs)
            })
            # Add all scores
            for score in SCORES:
                items[-1][score] = best_results[score]
        else:
            empty_folders.append(epoch_folder)

    if empty_folders:
        print('Empty folders')
        for f in empty_folders:
            print(f)
        print("")

    return items


def print_table(args, items, results_re_pattern, main_score):
    widths = []
    rows = [['experiment name', 'best epoch'] + SCORES]
    centering = ['<', '^'] + ['^'] * len(SCORES)
    for item in sorted(items, key=lambda x: x[main_score]):
        row = []
        row.append(item['results_folder'].split(args.checkpoints)[1][1:])
        row.append(
            '{}/{}'.format(item[f'best_{main_score}_epoch'],
            max(item['epochs']))
        )
        for score in SCORES:
            if item[score] is None:
                row.append(' ')
            else:
                row.append('{:0.1f}'.format(item[score]))
        rows.append(row)

    ptable(rows, centering)


def ptable(rows, centering):

    num_columns = len(rows[0])
    # bash scape chars (used for formatting, have length 0 on display)
    BASH_SCAPE = re.compile('\\x1b\[\d+m|\\x1b\[0m')
    column_widths = [max([len(BASH_SCAPE.sub('', row[i])) for row in rows]) for i in range(num_columns)]

    table_str = []
    col_sep = ' '
    for i,  row in enumerate(rows):
        row_str = []
        for j, cell in enumerate(row):
            # need to discount for bash scape chars
            delta = len(cell) - len(BASH_SCAPE.sub('', cell))
            if i == 0:
                # Header has all cells centered
                align = '^'
            else:    
                align = centering[j]
            row_str.append('{:{align}{width}} '.format(cell, align=align, width=column_widths[j] + delta))
        table_str.append(col_sep.join(row_str))
            
    row_sep = '\n'
    print(row_sep.join(table_str))
    print("")
    

if __name__ == '__main__':

    # ARGUMENT HANDLING
    args = argument_parsing()

    main_score = 'BLEU'

    # collect results for each model
    items = collect_results(args, results_re, main_score)

    if items == []:
        exit()

    # link best score model
    if args.link_best:
        for item in items:
            folder = item['model_folder']
            epoch = item[f'best_{main_score}_epoch']
            source_best = f'checkpoint_mymodel_{epoch}.pth'
            target_best = f'{folder}/checkpoint_best_{main_score}.pth'
            if (
                os.path.islink(target_best) and
                os.path.basename(os.path.realpath(target_best)) != 
                    source_best
            ):
                os.remove(target_best)
            if not os.path.islink(target_best):
                os.symlink(source_best, target_best)

    # print results
    if items != [] and not args.no_print:
        print_table(args, items, results_re.pattern, main_score)
